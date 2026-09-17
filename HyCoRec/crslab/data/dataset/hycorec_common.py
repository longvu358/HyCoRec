"""Shared conversation-processing helpers for the HyCoRec datasets
(``HReDial`` / ``HTGReDial``).

The two dataset classes were byte-for-byte identical in their conversation
processing, so the logic lives here once. It implements the CPC-Hypergraph v2
data contract (``docs/contexual_personal_collective/cpc_hypergraph_v2_method_spec.md``):

* **turn-local** ``item`` / ``entity`` / ``word`` lists per merged turn -- required
  for scope C (Contextual). The previous implementation shared one growing list
  across every turn dict, collapsing the per-turn structure.
* the raw ``list[list[conv]]`` per-user grouping is **kept** (not flattened):
  ``group[-1]`` is the target session, ``group[:-1]`` are the user's historical
  sessions ``D_u(d)`` -- required for scope P (Personal). This mirrors the
  original MHIM pipeline (``MHIM/crslab/data/dataset/hredial/hredial.py``).

Every rec/conv sample therefore carries:

* ``tokens`` / ``response``                       -- unchanged, token ids
* ``item`` / ``entity`` / ``word``                -- per-turn lists (turn-local), history < current turn
* ``item_global`` / ``entity_global`` / ``word_global`` -- flat V_f(d<=t) accumulators for readout
* ``history_session_items`` / ``..._entities`` / ``..._words`` -- list of per-session node sets (scope P)
* ``items``                                       -- current turn's gold items (rec targets)
* ``conv_id`` / ``user_id``
"""

from copy import copy


def merge_conv_turns(dialog, tok2ind, entity2id, unk_idx):
    """Merge consecutive same-role utterances into turns with **turn-local** ids.

    Returns a list of dicts ``{role, text, item, entity, word}`` where each list
    holds only the ids mentioned in that turn (matching the original CRSLab
    ``ReDialDataset._merge_conv_data``).
    """
    turns = []
    last_role = None
    for utt in dialog:
        text_token_ids = [tok2ind.get(w, unk_idx) for w in utt["text"]]
        item_ids = [entity2id[m] for m in utt["movies"] if m in entity2id]
        entity_ids = [entity2id[e] for e in utt["entity"] if e in entity2id]
        word_ids = [tok2ind[w] for w in utt["text"] if w in tok2ind]

        if utt["role"] == last_role:
            turns[-1]["text"] += text_token_ids
            turns[-1]["item"] += item_ids
            turns[-1]["entity"] += entity_ids
            turns[-1]["word"] += word_ids
        else:
            turns.append(
                {
                    "role": utt["role"],
                    "text": text_token_ids,
                    "item": item_ids,
                    "entity": entity_ids,
                    "word": word_ids,
                }
            )
        last_role = utt["role"]
    return turns


def session_node_sets(turns):
    """Collapse a whole session's turns into ``(items, entities, words)``.

    ``entities`` includes items (same id space, as in HyCoRec). Used to build the
    per-session hyperedges of scope P.
    """
    items, entities, words = set(), set(), set()
    for turn in turns:
        items.update(turn["item"])
        entities.update(turn["entity"])
        entities.update(turn["item"])
        words.update(turn["word"])
    return sorted(items), sorted(entities), sorted(words)


def build_history_sessions(history_turns, k_hist):
    """From the user's historical merged sessions build the scope-P node sets.

    Keeps only the most recent ``k_hist`` sessions and only sessions that mention
    at least one item (``H^P_I`` is seeded from items).
    """
    sessions = [session_node_sets(turns) for turns in history_turns]
    sessions = [s for s in sessions if len(s[0]) > 0]
    if k_hist is not None and k_hist > 0:
        sessions = sessions[-k_hist:]
    return sessions


def augment_target_session(target_turns, conv_id, user_id, history_sessions):
    """Generate one sample per turn (after the first) of the target session.

    ``history_sessions`` is shared by reference across all samples of the same
    conversation -- it is read-only downstream (the MHIM hyperedge-extension step
    that used to mutate it is disabled).
    """
    samples = []
    context_tokens, context_items, context_entities, context_words = [], [], [], []

    global_items = []
    global_entities, entity_seen = [], set()
    global_words, word_seen = [], set()

    hist_items = [s[0] for s in history_sessions]
    hist_entities = [s[1] for s in history_sessions]
    hist_words = [s[2] for s in history_sessions]

    for turn in target_turns:
        text_tokens = turn["text"]
        entities, movies, words = turn["entity"], turn["item"], turn["word"]

        if len(context_tokens) > 0:
            samples.append(
                {
                    "conv_id": conv_id,
                    "user_id": user_id,
                    "role": turn["role"],
                    "tokens": copy(context_tokens),
                    "response": text_tokens,
                    # scope C: genuine per-turn (turn-local) node lists, turns <= current
                    "item": copy(context_items),
                    "entity": copy(context_entities),
                    "word": copy(context_words),
                    # readout V_f(d<=t)
                    "item_global": list(global_items),
                    "entity_global": list(global_entities),
                    "word_global": list(global_words),
                    # scope P: per-session historical node sets (shared ref, read-only)
                    "history_session_items": hist_items,
                    "history_session_entities": hist_entities,
                    "history_session_words": hist_words,
                    # rec target: current turn's items only (turn-local)
                    "items": movies,
                }
            )

        context_tokens.append(text_tokens)
        context_items.append(movies)
        context_entities.append(entities + movies)
        context_words.append(words)

        global_items += movies
        for entity in entities + movies:
            if entity not in entity_seen:
                entity_seen.add(entity)
                global_entities.append(entity)
        for word in words:
            if word not in word_seen:
                word_seen.add(word)
                global_words.append(word)

    return samples


def process_grouped_raw_data(raw_data, tok2ind, entity2id, unk_idx, k_hist):
    """Full ``_raw_data_process`` body shared by both HyCoRec datasets.

    ``raw_data`` is ``list[list[conv]]`` grouped by user; the last conv of each
    group is the target session, the earlier ones are ``D_u(d)``.
    """
    augmented = []
    for group in raw_data:
        merged = [
            merge_conv_turns(conv["dialog"], tok2ind, entity2id, unk_idx)
            for conv in group
        ]
        target_turns = merged[-1]
        target_conv = group[-1]
        history_sessions = build_history_sessions(merged[:-1], k_hist)
        conv_id = int(target_conv["conv_id"])
        user_id = target_conv.get("user_id")
        augmented.extend(
            augment_target_session(target_turns, conv_id, user_id, history_sessions)
        )
    return augmented
