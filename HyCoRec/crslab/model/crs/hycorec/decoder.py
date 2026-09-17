import torch
import numpy as np
from torch import nn
from crslab.model.utils.modules.transformer import MultiHeadAttention, TransformerFFN, _normalize, create_position_codes


class TransformerDecoderLayerKG(nn.Module):
    def __init__(
            self,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm_self_attention = nn.LayerNorm(embedding_size)

        self.session_item_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm_session_item_attention = nn.LayerNorm(embedding_size)

        self.related_encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm_related_encoder_attention = nn.LayerNorm(embedding_size)

        self.context_encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm_context_encoder_attention = nn.LayerNorm(embedding_size)

        self.norm_merge = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, related_encoder_output, related_encoder_mask, context_encoder_output, context_encoder_mask,
                session_embedding, session_mask):
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm_self_attention)

        residual = x
        x = self.session_item_attention(
            query=x,
            key=session_embedding,
            value=session_embedding,
            mask=session_mask
        )
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm_session_item_attention)

        residual = x
        related_x = self.related_encoder_attention(
            query=x,
            key=related_encoder_output,
            value=related_encoder_output,
            mask=related_encoder_mask
        )
        related_x = self.dropout(related_x)  # --dropout

        context_x = self.context_encoder_attention(
            query=x,
            key=context_encoder_output,
            value=context_encoder_output,
            mask=context_encoder_mask
        )
        context_x = self.dropout(context_x)  # --dropout

        x = related_x * 0.1 + context_x * 0.9 + residual
        x = _normalize(x, self.norm_merge)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def forward_incremental(self, x, related_encoder_output, related_encoder_mask, context_encoder_output,
                             context_encoder_mask, session_embedding, session_mask, incr_state):
        """Single-new-token variant of ``forward`` - see
        ``MultiHeadAttention.forward_incremental``. ``x`` is just the newest
        token (``[B, 1, dim]``); ``incr_state`` is this layer's cache from the
        previous step (``None`` on the first step). Returns ``(new_x,
        new_incr_state)``.
        """
        if incr_state is None:
            self_state = session_state = related_state = context_state = None
        else:
            self_state, session_state, related_state, context_state = incr_state

        residual = x
        x, self_state = self.self_attention.forward_incremental(
            query=x, incr_state=self_state, static_kv=False
        )
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm_self_attention)

        residual = x
        x, session_state = self.session_item_attention.forward_incremental(
            query=x,
            key=session_embedding,
            value=session_embedding,
            mask=session_mask,
            incr_state=session_state,
            static_kv=True,
        )
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm_session_item_attention)

        residual = x
        related_x, related_state = self.related_encoder_attention.forward_incremental(
            query=x,
            key=related_encoder_output,
            value=related_encoder_output,
            mask=related_encoder_mask,
            incr_state=related_state,
            static_kv=True,
        )
        related_x = self.dropout(related_x)  # --dropout

        context_x, context_state = self.context_encoder_attention.forward_incremental(
            query=x,
            key=context_encoder_output,
            value=context_encoder_output,
            mask=context_encoder_mask,
            incr_state=context_state,
            static_kv=True,
        )
        context_x = self.dropout(context_x)  # --dropout

        x = related_x * 0.1 + context_x * 0.9 + residual
        x = _normalize(x, self.norm_merge)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        new_incr_state = (self_state, session_state, related_state, context_state)
        return x, new_incr_state

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class TransformerDecoderKG(nn.Module):
    """
    Transformer Decoder layer.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
            self,
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            vocabulary_size,
            embedding=None,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            embeddings_scale=True,
            learn_positional_embeddings=False,
            padding_idx=None,
            n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayerKG(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input, related_encoder_state, context_encoder_state, session_state, incr_state=None):
        related_encoder_output, related_encoder_mask = related_encoder_state
        context_encoder_output, context_encoder_mask = context_encoder_state
        session_embedding, session_mask = session_state

        seq_len = input.shape[1]
        positions = input.new_empty(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)  # (batch, seq_len)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        for layer in self.layers:
            tensor = layer(tensor, related_encoder_output, related_encoder_mask, context_encoder_output,
                           context_encoder_mask, session_embedding, session_mask)

        return tensor, None

    def forward_incremental(self, input, related_encoder_state, context_encoder_state, session_state, incr_state=None):
        """KV-cached variant of ``forward`` for greedy/autoregressive decoding.

        Unlike ``forward``, ``input`` must be exactly the single newest token
        (``[B, 1]``) on every call, including the first (pass the start token
        alone, not a growing sequence). ``incr_state`` is the state returned
        by the previous call (``None`` for the first call).

        ``forward`` reprocesses the entire sequence generated so far - full
        self-attention plus all three cross-attentions - on every single
        step, which is O(current length) of wasted recomputation per step
        (~25x more attention work than necessary over a typical greedy
        decode, measured empirically). This does only the O(1) new work per
        step instead, by caching self-attention K/V (extended each step) and
        cross-attention K/V (fixed encoder states, computed once and
        reused). Produces numerically identical logits to calling ``forward``
        on the full sequence-so-far and slicing out the last position -
        verified via ``torch.allclose``.

        Returns ``(new_token_tensor, new_incr_state)``.
        """
        related_encoder_output, related_encoder_mask = related_encoder_state
        context_encoder_output, context_encoder_mask = context_encoder_state
        session_embedding, session_mask = session_state

        assert input.shape[1] == 1, 'forward_incremental expects exactly one new token per call'
        if incr_state is None:
            offset = 0
            layer_states = [None] * self.n_layers
        else:
            offset, layer_states = incr_state

        position = input.new_full((1, 1), offset).long()
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(position).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        new_layer_states = []
        for layer, layer_state in zip(self.layers, layer_states):
            tensor, layer_state = layer.forward_incremental(
                tensor, related_encoder_output, related_encoder_mask, context_encoder_output,
                context_encoder_mask, session_embedding, session_mask, layer_state,
            )
            new_layer_states.append(layer_state)

        return tensor, (offset + 1, new_layer_states)
