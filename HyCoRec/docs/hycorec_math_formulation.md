# Mô hình toán học của HyCoRec

Tài liệu này mô tả công thức toán học của `HyCoRecModel`
(`HyCoRec/crslab/model/crs/hycorec/hycorec.py`), bám sát đúng luồng tính toán
trong code (không suy diễn thêm phần không có trong triển khai).

Mô hình gồm 4 khối chính:

1. Mã hoá đồ thị tri thức (KG Encoding) bằng R-GCN.
2. Xây dựng siêu đồ thị (hyperedge construction) theo phiên hội thoại, ở hai
   phạm vi *global* (toàn bộ lịch sử) và *local* (cửa sổ `k` lượt thoại gần
   nhất).
3. Tích chập siêu đồ thị (Hypergraph Convolution) + hợp nhất bằng attention để
   ra biểu diễn người dùng (user representation) và biểu diễn phiên (session
   representation).
4. Module sinh hội thoại (Transformer encoder–decoder) có cơ chế copy và
   "thiên kiến" (bias) theo hồ sơ người dùng, cùng module gợi ý (recommendation).

---

## 1. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $\mathcal{G}=(\mathcal{E}_{kg}, \mathcal{R})$ | Đồ thị tri thức (entity-relation KG), cạnh $(h,r,t)$ |
| $n_e$ | Số thực thể (entity) trong KG |
| $d$ | `kg_emb_dim` — chiều embedding KG |
| $\mathbf{X}^{(0)}\in\mathbb{R}^{n_e\times d}$ | Ma trận embedding thực thể khởi tạo (`entity_embedding.weight`) |
| $\mathbf{X}^{(0)}_w\in\mathbb{R}^{n_e\times d}$ | Ma trận embedding "word" khởi tạo (`word_embedding.weight`) |
| $R,\ R_c,\ R_w$ | Tập item / entity / word được nhắc tới trong phiên hội thoại (đến lượt hiện tại) |
| $R^{loc},\dots$ | Tập tương ứng chỉ trong cửa sổ $k$ lượt gần nhất (`hyperedge_window_k`) |
| $\mathrm{Adj}(\cdot)$ | Hàng xóm 1-hop trong KG phụ trợ **tĩnh, toàn corpus**, tra từ `item_adj`/`entity_adj`/`word_adj` — xem mục 3 |
| $\mathbf{H}\in\{0,1\}^{N\times M}$ | Ma trận liên thuộc (incidence) của siêu đồ thị: $N$ nút, $M$ siêu cạnh |
| $D_v,\ B_e$ | Bậc nút $v$ và bậc (kích thước) siêu cạnh $e$ |

---

## 2. Mã hoá đồ thị tri thức (KG Encoding)

Ba bộ mã hoá R-GCN **độc lập trọng số** (`item_encoder`, `entity_encoder`,
`word_encoder`), cùng kiến trúc, áp dụng **một lớp** R-GCN (phân rã cơ sở —
basis decomposition, `num_bases = B`) lên toàn bộ đồ thị tri thức tĩnh
$(\texttt{edge\_idx}, \texttt{edge\_type})$:

$$
\mathbf{Z}^{item} = \mathrm{RGCN}_{item}(\mathbf{X}^{(0)}),\qquad
\mathbf{Z}^{ent} = \mathrm{RGCN}_{ent}(\mathbf{X}^{(0)}),\qquad
\mathbf{Z}^{word} = \mathrm{RGCN}_{word}(\mathbf{X}^{(0)}_w)
$$

với, cho mỗi thực thể $v$:

$$
z_v = \sigma\!\left(\mathbf{W}_0 x_v + \sum_{r\in\mathcal{R}}\sum_{u\in\mathcal{N}_r(v)}\frac{1}{c_{v,r}}\mathbf{W}_r x_u\right),
\qquad
\mathbf{W}_r=\sum_{b=1}^{B} a_{rb}\,\mathbf{V}_b
$$

($c_{v,r}=|\mathcal{N}_r(v)|$, chuẩn hoá theo bậc quan hệ; $\mathbf{V}_b$ là các
ma trận cơ sở dùng chung giữa các quan hệ). $\mathbf{Z}^{item},\mathbf{Z}^{ent},\mathbf{Z}^{word}\in\mathbb{R}^{n_e\times d}$ là embedding KG dùng làm đầu vào cho bước siêu đồ thị.

---

## 3. Xây dựng siêu đồ thị theo phiên (`_get_hypergraph`)

### 3.0 Nguồn của $\mathrm{Adj}(\cdot)$ — quan trọng, dễ hiểu nhầm

$\mathrm{Adj}(\cdot)$ **không phải** thống kê đồng-xuất-hiện trong hội thoại,
và **không phải** "các item khác trong cùng phiên". Nó được nạp một lần lúc
`build_model` (`_build_adjacent_matrix`, đọc từ `data/edger/<dataset>/*.pkl`)
và **cố định trong suốt huấn luyện/suy luận**, xây offline bởi `run_edger.py`
+ `edger/<dataset>.py` từ một **đồ thị tri thức phụ trợ, tĩnh, toàn corpus**:

- `item_adj = entity_adj`: hàng xóm 1-hop trong đồ thị con DBpedia
  (`dbpedia_subkg.json` cho ReDial/OpenDialKG, CN-DBpedia cho TGReDial,
  `entity_subkg.txt` cho DuRecDial) — với mỗi cạnh $(a,b)$ của KG này, $a$ và
  $b$ được thêm vào tập hàng xóm của nhau (đối xứng hoá đồ thị có hướng).
- `word_adj`: hàng xóm trong **ConceptNet** (`en_side.txt`/`zh_side.txt`),
  lọc theo vocab của dataset — tức quan hệ ngữ nghĩa từ vựng.

Vai trò của **phiên hội thoại** chỉ là chọn ra **tập hạt giống** $R_\bullet$
(item/entity/word được nhắc tới đến lượt hiện tại) — session quyết định *tâm*
của mỗi siêu cạnh, còn *hàng xóm mở rộng quanh tâm đó* luôn tra từ KG/ConceptNet
tĩnh nói trên, không phụ thuộc phiên nào cả.

### 3.1 Công thức xây siêu cạnh

Với mỗi mẫu (phiên hội thoại) và mỗi nhánh $\bullet\in\{item, entity, word\}$,
gọi $R_\bullet=\{v_1,\dots,v_k\}$ là tập các nút "hạt giống" (được nhắc tới
trong hội thoại, đã khử trùng lặp). Với mỗi hạt giống $v_i$, một **siêu cạnh**
được tạo bằng chính nó và hàng xóm 1-hop của nó trong KG phụ trợ tĩnh:

$$
e_i \;=\; \{v_i\}\ \cup\ \mathrm{Adj}(v_i), \qquad i=1,\dots,|R_\bullet|
$$

Tập nút của siêu đồ thị là hợp của mọi siêu cạnh:
$V_\bullet=\bigcup_i e_i$, và ma trận liên thuộc

$$
\mathbf{H}\in\{0,1\}^{|V_\bullet|\times|R_\bullet|},\qquad
H_{v,i}=\mathbb{1}[v\in e_i]
$$

Nhánh **local** dùng cùng công thức nhưng với $R_\bullet^{loc}\subseteq
R_\bullet$ giới hạn trong $k$ lượt thoại gần nhất, và có **bộ trọng số tích
chập riêng** (không chia sẻ với nhánh global) — xem mục 4.

---

## 4. Tích chập siêu đồ thị (`CustomHypergraphConv`)

Đầu vào là embedding con $\mathbf{X}_{sub}=\mathbf{Z}[V_\bullet]\in\mathbb{R}^{|V_\bullet|\times d}$
(lấy hàng từ $\mathbf{Z}^{item}/\mathbf{Z}^{ent}/\mathbf{Z}^{word}$ tuỳ nhánh).
Phép tích chập (không dùng attention, `use_attention=False`) là tích chập siêu
đồ thị chuẩn hai giai đoạn (kiểu Feng *et al.*, HGNN):

**Phép chiếu tuyến tính:**
$$
\mathbf{X}' = \mathbf{X}_{sub}\,\Theta
$$

**Bậc nút và bậc siêu cạnh** (không trọng số hoá siêu cạnh, $w_e\equiv1$):
$$
D_v = \sum_{e}H_{v,e}\quad(\text{số siêu cạnh chứa } v),\qquad
B_e = \sum_{v}H_{v,e}\quad(\text{kích thước siêu cạnh } e=|e|)
$$

**Giai đoạn 1 — nút → siêu cạnh** (trung bình các nút trong mỗi siêu cạnh):
$$
y_e = \frac{1}{B_e}\sum_{v\in e} x'_v
$$

**Giai đoạn 2 — siêu cạnh → nút** (trung bình các siêu cạnh chứa nút):
$$
z_v = \frac{1}{D_v}\sum_{e\ni v} y_e \;+\; b
$$

Viết dưới dạng ma trận (đúng công thức Laplacian siêu đồ thị chuẩn):

$$
\boxed{\;\mathbf{Z} = \mathbf{D}^{-1}\mathbf{H}\,\mathbf{B}^{-1}\mathbf{H}^{\top}\,\mathbf{X}_{sub}\,\Theta \;+\; \mathbf{b}\;}
$$

với $\mathbf{D}=\mathrm{diag}(D_v)$, $\mathbf{B}=\mathrm{diag}(B_e)$.

Có **6 bộ trọng số** $\Theta$ độc lập (mỗi bộ là một `CustomHypergraphConv`
riêng): `hyper_conv_item`, `hyper_conv_entity`, `hyper_conv_word` (global) và
`hyper_conv_item_local`, `hyper_conv_entity_local`, `hyper_conv_word_local`
(local, chỉ tồn tại khi `hyperedge_window_k` được đặt).

> Ghi chú cài đặt: `_batched_hyperconv` gộp nhiều mẫu trong batch thành một
> đồ thị hợp rời rạc (disjoint-union, kiểu `torch_geometric.data.Batch`) để
> gọi tích chập một lần, nhưng **về mặt toán học tương đương tuyệt đối** với
> gọi công thức trên riêng lẻ cho từng mẫu.

---

## 5. Hợp nhất bằng attention → biểu diễn người dùng (`encode_user`, `_attention_and_gating`)

Gọi $\mathbf{Z}^{it}_i,\mathbf{Z}^{en}_i,\mathbf{Z}^{wd}_i$ là các ma trận nút
đầu ra tích chập siêu đồ thị (global) của mẫu $i$ cho ba nhánh item/entity/word
(và tương tự $\mathbf{Z}^{it,loc}_i,\dots$ cho nhánh local nếu bật). Tập tri
thức tổng hợp của mẫu (mỗi hàng là một vector, ghép theo hàng):

$$
\mathcal{R}_i = \big[\mathbf{Z}^{it}_i;\ \mathbf{Z}^{en}_i;\ \mathbf{Z}^{wd}_i\ (;\ \mathbf{Z}^{it,loc}_i;\ \mathbf{Z}^{en,loc}_i;\ \mathbf{Z}^{wd,loc}_i)\big]\in\mathbb{R}^{M_i\times d}
$$

Gọi $\mathcal{C}_i = \mathbf{Z}^{ent}[R_{c,i}]$ (embedding KG thô — **chưa**
qua tích chập siêu đồ thị — của các entity được nhắc tới) là tập ngữ cảnh
(context set).

**Attention đa đầu chéo** (`MHItemAttention`, query = ngữ cảnh, key=value = tri thức):
$$
\tilde{\mathcal{C}}_i = \mathrm{MHA}(Q=\mathcal{C}_i,\ K=V=\mathcal{R}_i)
= \Big[\mathrm{concat}_h\big(\mathrm{softmax}(Q_hK_h^\top/\sqrt{d_h})V_h\big)\Big]W_O
$$
(mỗi hàng ngữ cảnh nhận một vector được "tưới" thông tin từ toàn bộ $\mathcal{R}_i$).

**Pooling tập → vector** dùng self-attention cộng tính (`SelfAttentionBatch`):
$$
\mathrm{Pool}(\mathbf{h}_1,\dots,\mathbf{h}_n) = \sum_{j=1}^n \alpha_j h_j,\qquad
\alpha_j=\mathrm{softmax}_j\big(\mathbf{b}^\top\tanh(\mathbf{A}^\top h_j)\big)
$$

Biểu diễn người dùng cuối cùng $u_i$ phụ thuộc `pooling` (`Attn` hoặc `Mean`)
và việc có ngữ cảnh hay không:

- **Không có ngữ cảnh** ($R_{c,i}=\emptyset$):
$$
u_i=\begin{cases}\mathrm{Pool}(\mathcal{R}_i) & \text{pooling=Attn}\\[2pt]
\mathrm{mean}(\mathcal{R}_i) & \text{pooling=Mean}\end{cases}
$$

- **Có ngữ cảnh**, đặt $\tilde{u}_i=\mathrm{Pool}(\tilde{\mathcal{C}}_i)$ (hoặc
  $\mathrm{mean}(\tilde{\mathcal{C}}_i)$ nếu Mean), rồi ghép vào tập ngữ cảnh và
  gộp lần nữa:
$$
u_i=\begin{cases}
\mathrm{Pool}\big([\mathcal{C}_i;\ \tilde u_i]\big) & \text{pooling=Attn}\\[4pt]
\mathrm{mean}\big([\mathcal{C}_i;\ \tilde u_i]\big) & \text{pooling=Mean}
\end{cases}
$$

Kết quả $\mathbf{u}\in\mathbb{R}^{B\times d}$ (`user_embedding`).

---

## 6. Biểu diễn phiên hội thoại (`encode_session`)

Không pooling — mọi vector nút được **giữ nguyên dạng chuỗi** (ghép theo trục
thời gian, đệm 0 bên trái đến độ dài lớn nhất trong batch):

$$
S_i=\big[\mathbf{Z}^{it}_i;\ \mathbf{Z}^{en}_i;\ (\mathcal{C}_i);\ \mathbf{Z}^{wd}_i\ (;\ \mathbf{Z}^{it,loc}_i;\mathbf{Z}^{en,loc}_i;\mathbf{Z}^{wd,loc}_i)\big]\in\mathbb{R}^{L_i\times d}
$$

$$
\mathbf{S} = \mathrm{PadLeft}(S_1,\dots,S_B)\in\mathbb{R}^{B\times L\times d},\qquad
\mathbf{S}^{tok} = \mathbf{S}\,\mathbf{W}_{e\to t}\in\mathbb{R}^{B\times L\times d_{tok}}
$$

($\mathbf{W}_{e\to t}$ = `entity_to_token`, ánh xạ không gian KG → không gian
token), cùng mặt nạ (mask) $\mathbf{M}\in\{0,1\}^{B\times L}$ đánh dấu vị trí
thật (không phải đệm).

---

## 7. Module gợi ý (Recommendation, `recommend`)

$$
\text{score}_i = \mathbf{Z}^{ent}\,u_i + \mathbf{b}_{rec}\ \in\mathbb{R}^{n_e}
\qquad\Longrightarrow\qquad
\mathcal{L}_{rec} = \frac{1}{B}\sum_{i=1}^B -\log\frac{\exp(\text{score}_i[y_i])}{\sum_{v}\exp(\text{score}_i[v])}
$$

($y_i$ = item mục tiêu; điểm số là tích vô hướng giữa biểu diễn người dùng và
embedding KG của mọi thực thể, cộng bias học được — dạng "dot-product +
bias" của recommender KG cổ điển).

---

## 8. Module sinh hội thoại (Conversation)

### 8.1 Encoder

Hai `TransformerEncoder` độc lập trọng số (kiến trúc self-attention + FFN
chuẩn) mã hoá chuỗi token liên quan (`related_tokens`) và ngữ cảnh hội thoại
(`context_tokens`):

$$
(\mathbf{H}^{rel},\mathbf{m}^{rel}) = \mathrm{Enc}_{rel}(\text{related\_tokens}),\qquad
(\mathbf{H}^{ctx},\mathbf{m}^{ctx}) = \mathrm{Enc}_{ctx}(\text{context\_tokens})
$$

### 8.2 Decoder (`TransformerDecoderKG`)

Với chuỗi đích dịch phải một bước (teacher forcing) $y_{<t}$, tại mỗi lớp
decoder, mỗi bước $t$:

**(a) Self-attention nhân quả (causal):**
$$
x^{(1)} = \mathrm{LN}\big(x + \mathrm{MHA}_{self}(x,x,x;\ \text{mask}_{causal})\big)
$$

**(b) Cross-attention vào biểu diễn phiên $\mathbf{S}^{tok}$:**
$$
x^{(2)} = \mathrm{LN}\big(x^{(1)} + \mathrm{MHA}_{sess}(x^{(1)}, \mathbf{S}^{tok}, \mathbf{S}^{tok};\ \mathbf{M})\big)
$$

**(c) Hai cross-attention song song vào $\mathbf{H}^{rel}$ và $\mathbf{H}^{ctx}$, hợp bằng tổng có trọng số cố định:**
$$
x_{rel} = \mathrm{MHA}_{rel}(x^{(2)},\mathbf{H}^{rel},\mathbf{H}^{rel};\mathbf{m}^{rel}),\qquad
x_{ctx} = \mathrm{MHA}_{ctx}(x^{(2)},\mathbf{H}^{ctx},\mathbf{H}^{ctx};\mathbf{m}^{ctx})
$$
$$
x^{(3)} = \mathrm{LN}\big(0.1\cdot x_{rel} + 0.9\cdot x_{ctx} + x^{(2)}\big)
$$

**(d) FFN:**
$$
\ell = \mathrm{LN}\big(x^{(3)} + \mathrm{FFN}(x^{(3)})\big)
$$

(lặp lại $n\_layers$ lớp; $\ell_t$ = trạng thái ẩn cuối cùng tại bước $t$).

### 8.3 Đầu ra: kết hợp ba nguồn logit (language model + user bias + copy)

$$
\text{logit}^{tok}_t = \ell_t\,\mathbf{E}_{tok}^{\top} \qquad\text{(chia sẻ trọng số embedding token)}
$$

$$
\text{logit}^{user} = \mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1\, u) \qquad\text{(hằng theo } t\text{, "thiên kiến" từ vựng theo hồ sơ người dùng/gợi ý)}
$$

$$
\text{copy\_latent}_t = \big[\mathbf{W}_{e\to t}\,u\ ;\ \ell_t\big],\qquad
\text{logit}^{copy}_t = \mathbf{W}_4\,\mathrm{ReLU}(\mathbf{W}_3\,\text{copy\_latent}_t)\ \odot\ \mathbf{m}_{copy}
$$

($\mathbf{m}_{copy}$ = mặt nạ giới hạn chỉ copy các token thực thể "@..."
— chỉ áp dụng cho `HReDial`).

**Phân bố từ vựng cuối cùng** (tổng cộng dồn ba logit, không softmax gate):
$$
\text{logit}_t = \text{logit}^{tok}_t + \text{logit}^{user} + \text{logit}^{copy}_t,\qquad
p(y_t\mid y_{<t}) = \mathrm{softmax}(\text{logit}_t)
$$

**Huấn luyện (teacher forcing)** — cross-entropy theo token, bỏ qua padding:
$$
\mathcal{L}_{conv} = \frac{1}{\sum_i |y_i|}\sum_{i,t} -\log p_\theta(y_{i,t}\mid y_{i,<t},\ \mathbf{S}_i,\mathbf{H}^{rel}_i,\mathbf{H}^{ctx}_i,u_i)
$$

**Sinh (inference)** — giải mã tham lam (greedy) tuần tự, tái sử dụng
key/value đã cache (không đổi công thức toán, chỉ là biến thể tính toán tăng
tốc, tương đương số học với công thức trên):
$$
\hat y_t = \arg\max_v\ p_\theta(v\mid \hat y_{<t},\dots),\quad t=1,\dots,\texttt{longest\_label}
$$

---

## 9. Mục tiêu huấn luyện tổng thể

Hai giai đoạn huấn luyện tách biệt (`stage ∈ {rec, conv}` trong `forward`),
mỗi giai đoạn tối ưu một hàm mất mát riêng trên cùng biểu diễn KG/siêu đồ thị
đã mô tả:

$$
\min_\theta \ \mathcal{L}_{rec}(\theta)\qquad\text{hoặc}\qquad \min_\theta \ \mathcal{L}_{conv}(\theta)
$$

không có trọng số kết hợp $\lambda$ giữa hai loss vì chúng được huấn luyện ở
hai pha khác nhau, không cộng gộp trong cùng một lượt cập nhật.
