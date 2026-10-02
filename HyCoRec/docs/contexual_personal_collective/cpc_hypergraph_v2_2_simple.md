# CPC-Hypergraph v2.2 — Tổng quan phương pháp (fusion theo HyCoRec)

Bản này thay v2.1. Dựng hypergraph (§2) giữ nguyên; **toàn bộ phần fusion sau hypergraph encoder (§3) được viết lại theo cơ chế attention với query $P_c$ của `_attention_and_gating` trong mã nguồn HyCoRec** (đã đọc code, chưa chạy). Khác code ở một điểm: G là một hypergraph thống nhất nên key của MHA là **7 vector đã pooling**, không phải tập node. Dùng để viết phần Method; đặc tả kỹ thuật cần đồng bộ theo §7.

## 0. Thay đổi so với v2.1

| | v2.1 | v2.2 |
|---|---|---|
| Hypergraph G | 3 hypergraph theo trường (I, E, W) | **1 hypergraph thống nhất** trên $\mathcal V_E\cup\mathcal V_W$, 1 HConv |
| Readout | mean mỗi scope × trường → $\mathbf p^s_f$ (9 vector) | C, P: mean theo trường (6 vector); G: mean trên toàn $\mathcal Q^G$ (1 vector) → **7 hàng** |
| Trộn scope | $\alpha_{f}=\text{softmax}(\mathbf a_f)$, 9 tham số toàn cục | **MHA với query $\mathbf P_c$ trên 7 hàng**; trọng số scope phụ thuộc mẫu, không cần tham số $\alpha$ |
| Query hội thoại hiện tại | chỉ vào Pool cuối | **query của MHA** (như MHIM Eq. 7 và code HyCoRec) |
| Cold-start | mask gradient $a_{f,P}$ | hàng rỗng bị mask khỏi key; không cần mask gradient |
| Baseline A0 | "tái hiện bài báo HyCoRec" | **tái hiện code HyCoRec** (xem §0.2) |
| $\mathbf E_I$ cơ sở | $\mathbf X^{(0)}_I$ | embedding item **sau R-GCN** (đúng code) |
| Ablation fusion | A10 ($\alpha=\tfrac13$) | F1–F4 (§4) |

### 0.1 Lý do sửa
1. $\alpha$ toàn cục không phản ứng với hội thoại: đổi chủ đề giữa chừng vẫn dùng cùng tỉ lệ C/P/G. MHA với query $\mathbf P_c$ (như MHIM Eq. 7 và code HyCoRec) cho trọng số theo từng mẫu.
2. G bản chất là một hypergraph thống nhất (siêu cạnh hội thoại, review, aspect cùng sống trong một đồ thị), nên v2.1 tách G thành ba trường I/E/W là không đúng bản chất. v2.2 cho G một HConv và một vector.
3. Giữ query-attention của HyCoRec làm nền để chênh lệch giữa A0 và A9 quy về scope nhiều nhất có thể. Do key là 7 vector đã pool nên vẫn có một chênh lệch về fusion so với code; dòng A0′ và F3 (§4) tách riêng phần này.

### 0.2 Bài báo HyCoRec và code lệch nhau — chốt baseline theo code

| Mục | Bài báo (ACL 2024) | Code (`hycorec.py`) | CPC theo |
|---|---|---|---|
| Fusion nhánh gợi ý | $\text{Pool}([\text{Pool}(P_h);P_c])$, không attention | nối tập node (item, entity, word) → **MHA, query = entity hội thoại** → mean → nối với entity hội thoại → mean | code, nhưng key là 7 vector đã pool (§3.3) |
| Số lớp HConv | $L=2$, multi-head + head-pooling | 1 `HypergraphConv` mỗi trường, 1 head | quét $L\in\{1,2\}$; A0 ở $L=1$ |
| $P_r$ (review Transformer) | có | không thấy trong `model/crs/hycorec/` | tùy chọn (§3.3) |
| Item/entity/word hypergraph | item: siêu cạnh phiên | cả ba: siêu cạnh = node + láng giềng từ `edger` | ghi rõ trong bài |
| Điểm gợi ý | $\mathbf E_I$ không nói rõ | `F.linear(u, entity_encoder(emb))` + bias | code |
| Pooling | không nói rõ | cấu hình `Mean` (ReDial); `Attn` là biến thể | Mean |

**Cần xác minh trước khi chạy (xem §6):** `_build_adjacent_matrix` có vẻ gán lại danh sách láng giềng thành `[]` trước khi duyệt, khiến mọi `adj` rỗng; nguồn của `related_item/entity/word` (hội thoại hiện tại hay lịch sử user) tùy biến thể dataset.

### 0.3 MHIM và HyCoRec — scope

MHIM định nghĩa *historical dialogue sessions* là các hội thoại trước đó của cùng user; HyCoRec kế thừa pipeline.

| Thành phần | MHIM | HyCoRec | Scope trong CPC |
|---|---|---|---|
| Session / Item-hypergraph | mỗi phiên lịch sử = 1 siêu cạnh | tương tự (bài báo) | **P** |
| Entity-hypergraph | item lịch sử + $N$-hop DBpedia | tương tự, $k$-hop | **P** |
| Word-hypergraph | không có | item + $k$-hop ConceptNet | **P** |
| Hyperedge extension | truy hồi phiên user khác | không nêu | **G dạng truy hồi** (giảm R@10 trên ReDial, MHIM Table 3) |
| Hội thoại hiện tại | R-GCN → $N_C$, làm query vào $[N_S;N_K]$ | R-GCN → $P_c$, làm query (code) | **C: dựng hypergraph** |

Ba khoảng trống: (1) hội thoại hiện tại chưa có hypergraph ngữ cảnh → scope C; (2) không có cấu trúc corpus-level tĩnh → scope G; (3) review chưa vào hypergraph → review-hypergraph ở P và G.

### 0.4 HyFairCRS — review-hypergraph đã có nhưng khác
Với mỗi item trong phiên lịch sử, HyFairCRS tạo hai siêu cạnh $\{i\}\cup W^+(\mathcal R_i)$ và $\{i\}\cup W^-(\mathcal R_i)$, node là **word**. CPC khác ở ba điểm: node là **entity KG** (dùng chung $\mathbf W_E$ với entity-hypergraph); phạm vi gồm **toàn bộ $\mathcal V_I$** tĩnh ở scope G (item đuôi dài vẫn nhận tín hiệu); thêm chiều **aspect-centric** (entity → tập item) ở trường $I$.

### 0.5 Đóng góp so với baseline

| | HyCoRec | MHIM-ext | HyFairCRS | **CPC** |
|---|---|---|---|---|
| Hypergraph ngữ cảnh hiện tại | ✗ | ✗ | ✗ | **✓ scope C** |
| Corpus-level tĩnh | ✗ | truy hồi | ✗ | **✓ scope G** |
| Review-hypergraph, node entity KG | ✗ | ✗ | ✗ (word) | **✓ P + G** |
| Item đuôi dài nhận tín hiệu review | ✗ | ✗ | ✗ | **✓ G** |
| Aspect-centric (entity → item) | ✗ | ✗ | ✗ | **✓ G, trường I** |

---

## 1. Ký hiệu

- $\mathcal V_I\subseteq\mathcal V_E$, $\mathcal V_W$: tập item, entity DBpedia, word ConceptNet. Trường $f\in\{I,E,W\}$.
- Hội thoại $d$, lượt $t_s$; $\mathcal V_f(t_s)$ node trường $f$ tại lượt $s$; $\mathcal V_f(d_{\le t})=\bigcup_{s\le t}\mathcal V_f(t_s)$.
- $\mathcal D_u(d)$: các phiên trước của user $u$ (cắt $K_{\text{hist}}$ gần nhất); $\mathcal I^P=\bigcup_{d'\in\mathcal D_u(d)}\mathcal V_I(d')$.
- $\mathcal E^{+}_{\text{top}}(\mathcal R_i)$: top-$k_{\text{rev}}$ entity (không phải item) liên kết từ câu review **dương** của item $i$, sắp theo tần suất.
- $\text{tf}^{+}(c,i)$: số câu review dương của $i$ chứa entity $c$; $\text{df}(c)=|\{i: c\in\mathcal E^{+}_{\text{top}}(\mathcal R_i)\}|$.
- $\mathbf P_c\in\mathbb R^{n_c\times d}$: embedding R-GCN của các entity trong hội thoại hiện tại (knowledge-aspect của HyCoRec), $n_c=|\mathcal V_E(d_{\le t})|$.
- $\mathcal V^G=\mathcal V_E\cup\mathcal V_W$: không gian node thống nhất của G (chỉ số word cộng offset $n_E$); $\mathcal V_I\subseteq\mathcal V_E$ nên item là một tập con của $\mathcal V^G$.
- Incidence nhị phân; trọng số siêu cạnh $=1$.

---

## 2. Ba scope — dựng hypergraph (C và P giữ nguyên; G sửa thành một hypergraph thống nhất)

### 2.1 Scope Personal $\mathcal H^P$ — mở rộng HyCoRec với review

$$
\mathcal H^P_I=\{h_{d'}=\mathcal V_I(d'):d'\in\mathcal D_u(d)\},\quad
\mathcal H^P_W=\{h_i=\{i\}\cup\mathcal N^{(k)}_{\text{ConceptNet}}(i):i\in\mathcal I^P\}. \tag{1}
$$

**Review-hypergraph scope P**, bổ sung vào trường $E$:

$$
\mathcal H^{P,\text{rev}}_E=\{h^{\text{rev}}_i=\{i\}\cup\mathcal E^{+}_{\text{top}}(\mathcal R_i):i\in\mathcal I^P\}. \tag{2}
$$

$$
\mathbf N^P_E = \bigl[\,\mathbf N^{P,\text{KG}}_E\;\big|\;\mathbf N^{P,\text{rev}}_E\,\bigr]. \tag{3}
$$

Node của hai vế đều thuộc $\mathcal V_E$ nên dùng chung $\mathbf X^{(0)}_E$ và $\mathbf W_E$. Tắt hyperedge extension của MHIM nếu code kế thừa.

*Ghi chú theo code:* ở HyCoRec code, mỗi siêu cạnh trường $I$ cũng là "node + láng giềng `edger`" chứ không phải siêu cạnh phiên như Eq. (1). Chọn một cách và ghi rõ trong bài.

### 2.2 Scope Contextual $\mathcal H^C$ — cửa sổ trượt

Học quan hệ xuyên lượt ngắn hạn: nhắc "Inception" ở lượt 3, "Christopher Nolan" ở lượt 4; siêu cạnh theo từng lượt đơn lẻ không nối được hai node này.

**[Cách A — đang dùng]** tại mỗi $t'\le t$, một siêu cạnh gộp $w$ lượt liên tiếp:

$$
\mathcal H^C_f(d,t)=\Bigl\{h_{t'}=\bigcup_{s=\max(1,\,t'-w+1)}^{t'}\mathcal V_f(t_s):t'\le t,\ |h_{t'}|\ge2\Bigr\}. \tag{4}
$$

Không mở rộng $k$-hop, để C là tín hiệu co-occurrence ngắn hạn thuần túy, phân biệt với P là KG-expansion dài hạn. Khởi điểm $w=3$; quét $\{2,3,4\}$.

**[Cách B — để dành]** ghép siêu cạnh lượt đơn $h_s=\mathcal V_f(t_s)$ với siêu cạnh cửa sổ; cần ablation riêng.

### 2.3 Scope Collective $\mathcal H^G$ — một hypergraph thống nhất, tĩnh

Dựng một lần trên $\mathcal D_{\text{train}}$. Khác C và P (mỗi trường một hypergraph), G là **một** hypergraph trên $\mathcal V^G$ chứa cả ba loại siêu cạnh dưới đây trong cùng ma trận incidence.

**(a) Siêu cạnh hội thoại**, node gộp mọi loại:

$$
\mathcal H^{G,\text{dlg}}=\{h_d=\mathcal V_I(d)\cup\mathcal V_E(d)\cup\mathcal V_W(d):d\in\mathcal D_{\text{train}},\ |h_d|\ge2\}. \tag{5}
$$

Khác MHIM extension: không truy hồi theo query hiện tại, không giới hạn số lượng.

**(b) Review-hypergraph**, toàn bộ $\mathcal V_I$:

$$
\mathcal H^{G,\text{rev}}=\{h^{\text{rev}}_i=\{i\}\cup\mathcal E^{+}_{\text{top}}(\mathcal R_i):i\in\mathcal V_I\}. \tag{6}
$$

Item chưa từng được nhắc nhưng có review vẫn kết nối vào mạng.

**(c) Siêu cạnh aspect-centric**, chỉ chứa item:

$$
\mathcal C=\{c\in\mathcal V_E\setminus\mathcal V_I:\text{df}(c)\ge m_{\min}\},\qquad
h^{\text{asp}}_c=\text{Top-}k_{\text{asp}}\{i:c\in\mathcal E^{+}_{\text{top}}(\mathcal R_i)\}\ \text{theo tf}^{+}(c,i). \tag{7}
$$

Siêu cạnh nối trực tiếp các item cùng được review nhắc tới $c$; node $c$ không nằm trong siêu cạnh (tùy chọn: thêm $c$). Loại $c\in\mathcal V_I$ để tên phim trong review không tạo cạnh item–item trực tiếp.

$$
\mathbf N^G=[\mathbf N^{G,\text{dlg}}\mid\mathbf N^{G,\text{rev}}\mid\mathbf N^{G,\text{asp}}]\in\{0,1\}^{|\mathcal V^G|\times|\mathcal E^G|}. \tag{8}
$$

Tiền tính một ma trận thưa $\hat{\mathbf A}^G=(V^G)^{-1}N^G(E^G)^{-1}(N^G)^{\top}$; không bao giờ tạo dense.

*Lưu ý kích thước siêu cạnh:* $h_d$ gộp cả word nên có thể rất lớn; chuẩn hóa $(E^G)^{-1}$ sẽ làm loãng đóng góp mỗi node. In phân bố $|h|$ theo loại siêu cạnh trước khi huấn luyện, và cân nhắc giới hạn số word trong $h_d$.


---

## 3. Mô hình — viết lại

### 3.1 HConv

Scope C và P, theo từng trường $f\in\{I,E,W\}$:

$$
\mathbf X^{s,(l+1)}_f=(\mathbf V^s_f)^{-1}\mathbf N^s_f(\mathbf E^s_f)^{-1}(\mathbf N^s_f)^{\top}\mathbf X^{s,(l)}_f\mathbf W^{(l)}_f,\quad s\in\{C,P\}. \tag{9}
$$

Scope G, một lần cho cả đồ thị thống nhất:

$$
\mathbf X^{G,(l+1)}=\hat{\mathbf A}^G\,\mathbf X^{G,(l)}\,\mathbf W^{(l)}_G. \tag{9G}
$$

- Đầu vào: $\mathbf X^{s,(0)}_f=\mathbf X^{(0)}_f[\mathcal V^s_f]$; $\mathbf X^{G,(0)}$ gồm hàng entity (từ `entity_encoder`) và hàng word (từ `word_encoder`), cùng chiều $d$. $\mathbf X^{(0)}_f$ là đầu ra R-GCN như code.
- Trọng số: $\mathbf W_f$ dùng chung giữa C và P (hai scope nhỏ, tham số riêng sẽ overfit). Với G mặc định **buộc** $\mathbf W_G:=\mathbf W_E$, vì phần lớn node của G là entity/item nên G ở cùng không gian với trường $E$ của C và P. Phương án tách $\mathbf W_G$ là ablation A12. Chưa chắc buộc tham số là tốt cho hàng word của G; cần xem kết quả.
- Không phi tuyến giữa tầng; $L\in\{1,2\}$ (code dùng 1, bài báo nói 2).

### 3.2 Readout: bảy vector

C và P pooling theo trường; G pooling một lần trên toàn bộ node liên quan:

$$
\mathbf r^s_f=\frac{1}{|\mathcal Q^s_f|}\sum_{v\in\mathcal Q^s_f}\mathbf X^s_f[v]\quad(s\in\{C,P\}),\qquad
\mathbf r^G=\frac{1}{|\mathcal Q^G|}\sum_{v\in\mathcal Q^G}\mathbf X^G[v]. \tag{10}
$$

$$
\mathcal Q^C_f=\mathcal V_f(d_{\le t}),\qquad
\mathcal Q^P_f=\mathcal V(\mathcal H^P_f)\ (\text{toàn bộ node, như HyCoRec}),\qquad
\mathcal Q^G=\bigcup_{f}\bigl(\mathcal V_f(d_{\le t})\cup\mathcal V^P_f\bigr). \tag{10'}
$$

Scope G không đọc toàn đồ thị: nó lan truyền toàn cục rồi gather embedding đã làm giàu tại các node liên quan tới user. Mean trên $\mathcal Q^G$ trộn node thuộc nhiều loại, nên loại nhiều node nhất (thường là word) chi phối. Bản mặc định giữ mean đơn giản; biến thể cân bằng theo loại (mean mỗi loại rồi trung bình) là ablation A13.

### 3.3 Fusion: MHA với query hội thoại hiện tại

$$
\mathbf R=\bigl[\,\mathbf r^C_I;\mathbf r^C_E;\mathbf r^C_W;\ \mathbf r^P_I;\mathbf r^P_E;\mathbf r^P_W;\ \mathbf r^G\,\bigr]\in\mathbb R^{7\times d}, \tag{11}
$$

$$
\tilde{\mathbf N}=\text{MHA}\bigl(\mathbf P_c,\ \mathbf R,\ \mathbf R\bigr)\in\mathbb R^{n_c\times d}, \tag{12}
$$

$$
\mathbf u=\text{Pool}\bigl([\,\text{Pool}(\tilde{\mathbf N})\,;\,\mathbf P_c\,]\bigr). \tag{13}
$$

- MHA là `nn.MultiheadAttention(d, 4)`, dùng chung module và trọng số với `item_attn` của baseline; không thêm tham số.
- Pool = mean theo hàng (cấu hình `Mean`). Với Mean, Eq. (13) cho $\mathbf u=\frac{1}{n_c+1}\bigl(\text{mean}(\tilde{\mathbf N})+\sum_j\mathbf P_c[j]\bigr)$. Biến thể `Attn`: thay bằng `SelfAttentionBatch` như code.
- Trọng số scope **xuất hiện từ attention**: attention mass trên hàng $(s,f)$ là $\alpha_{s,f}(d,t)$ phụ thuộc mẫu. Ghi lại để phân tích định tính.
- **Hàng không có node** (ví dụ chưa có lịch sử nên không có hàng P, hoặc đầu hội thoại chưa có word) bị mask khỏi key, không chèn hàng 0. Code gốc chèn hàng 0, nên giữ hàng 0 nếu cần khớp số A0 tuyệt đối.

| Tình huống | Xử lý |
|---|---|
| $n_c=0$ | $\mathbf u=\text{mean}(\mathbf R)$ |
| mọi hàng rỗng | $\mathbf u=\text{mean}(\mathbf P_c)$ |
| cả hai rỗng | $\mathbf u=\mathbf 0$ |

**Pooling trước attention.** Mỗi hàng của $\mathbf R$ đã là trung bình, nên $\mathbf P_c$ chọn được scope/trường nhưng không chọn được node, khác code HyCoRec (key là từng node). Dòng F3 (§4) chạy bản node-level để đo mất mát này.

**Nhãn hàng (A10).** Các hàng trong $\mathbf R$ chỉ khác nhau qua nội dung, nên attention không có cách tường minh để ưu tiên một scope. Thêm $\tilde{\mathbf r}_k=\mathbf r_k+\mathbf e_k$, $\mathbf e_k\in\mathbb R^d$, $k=1..7$, khởi tạo 0 (tổng $7d$ tham số), để mô hình bắt đầu như bản không nhãn.

**Review Transformer $P_r$ (có điều kiện).** Code HyCoRec công khai không dùng $P_r$. Nếu codebase của bạn có, chiếu về $d$ chiều và thêm một hàng $\mathbf r^{P_r}$ vào $\mathbf R$ (thành 8 hàng); nếu không, A11 bị bỏ.

### 3.4 Điểm gợi ý

$$
P_{\text{rec}}=\text{softmax}\bigl(\mathbf u\,\mathbf E_I^{\top}+\mathbf b\bigr). \tag{14}
$$

- **(a) cơ sở:** $\mathbf E_I=\mathbf X^{R\text{-GCN}}_I$, embedding item sau R-GCN như code.
- **(b) bắt buộc so sánh:** $\mathbf E_I=\mathbf X^{R\text{-GCN}}_I+\mathbf X^G[\mathcal V_I]$ — item ứng viên nhận tín hiệu từ G. Đây là cơ chế duy nhất để $h^{\text{rev}}$ và $h^{\text{asp}}$ tác động lên item đuôi dài ở phía ứng viên. $\mathbf X^G$ đã tính trong (9G).

Nhánh sinh câu giữ nguyên HyCoRec, thay biểu diễn người dùng bằng $\mathbf u$.

---

## 4. Ablation

Ký hiệu: **P** = Personal như HyCoRec (không có review-hypergraph); **P+R** = P thêm Eq. (2,3); **G0** = G chỉ $h_d$ (Eq. 5); **G** = G đầy đủ (Eq. 8). Cấu hình không có scope nào thì hàng tương ứng không có trong $\mathbf R$.

| # | Cấu hình | $\mathbf E_I$ | Trả lời câu hỏi |
|---|---|---|---|
| A0 | P, $L=1$, fusion node-level như code | (a) | tái hiện code HyCoRec — phải khớp trước khi làm gì |
| A0′ | P, fusion 3 hàng (Eq. 11–13 chỉ với P) | (a) | đổi sang 7-hàng có tự nó đổi kết quả không |
| A1 | P + C ($w=3$) | (a) | cửa sổ trượt có ích không |
| A2 | P + C ($w=1$) | (a) | co-mention lượt đơn — đường cơ sở cho C |
| A3 | P + G0 | (a) | corpus tĩnh-toàn cục vs không có |
| A4 | P + MHIM-extension | (a) | corpus truy hồi — so trực tiếp với A3 |
| A5 | P+R | (a) | review-hypergraph scope P |
| A6 | P + G0 + $h^{\text{rev}}$ | (b) | review-hypergraph trong G, phía ứng viên |
| A7 | P + G0 + $h^{\text{asp}}$ | (b) | aspect-centric trong G, phía ứng viên |
| A8 | P+R + G | (b) | review đầy đủ ở P và G, aspect ở G |
| A9 | C + P+R + G | (b) | **full** |
| A10 | A9 + nhãn hàng $\mathbf e_k$ | (b) | attention có cần biết scope không |
| A11 | A9 thêm/bỏ $P_r$ (nếu codebase có) | (b) | $P_r$ và review-hypergraph redundant không |
| A12 | A9, $\mathbf W_G$ tách khỏi $\mathbf W_E$ | (b) | buộc tham số G–E có hại không |
| A13 | A9, $\mathbf r^G$ cân bằng theo loại node | (b) | word có chi phối readout G không |
| F1 | A9 không query: $\mathbf u=\text{mean}([\text{mean}(\mathbf R);\mathbf P_c])$ | (b) | query $\mathbf P_c$ đóng góp bao nhiêu |
| F2 | A9, $\alpha$ toàn cục học trên 7 hàng (v2.1) | (b) | trọng số theo mẫu vs toàn cục |
| F3 | A9, key là từng node (code-style) thay vì 7 hàng | (b) | pooling trước attention mất bao nhiêu |
| F4 | A9, pooling `Attn` | (b) | Mean vs Attn pooling |

Dòng A4 quan trọng nhất về phản biện ("G khác extension MHIM chỗ nào"): câu trả lời phải là số. A6, A7 kỳ vọng tăng rõ nhất ở Coverage@k và Tail-Recall@10. A0 → A0′ → A1 là chuỗi tách thay đổi fusion khỏi thay đổi scope. Cả MHIM (Table 3) lẫn HyCoRec (Table 4) đều không ablation cách fusion, nên F1–F4 là kết quả mới của bài, không có sẵn bằng chứng để trích.

**Phân tích định tính cần xuất:** attention mass trên 7 hàng, tách theo lượt $t$; Recall@10 theo nhóm $t$ (kỳ vọng lợi ích của C tăng theo $t$); số item đuôi dài được "cứu" bởi $h^{\text{asp}}$ cạnh Tail-Recall của A6 vs A5.

---

## 5. Để dành cho các lần sau

1. Decay $N(\lambda,w)$ trong C/P; decay giữa các phiên cho $N^P$.
2. Attention theo trường (MHA riêng mỗi trường, rồi attention giữa các trường).
3. Trọng số TF-IDF/cực tính cho $h^{\text{rev}}$ và $h^{\text{asp}}$.
4. Siêu cạnh đồng xuất hiện SPPMI + k-truss trong G.
5. Cross-scope contrastive $C\leftrightarrow G$ (view = scope, khác HyFairCRS).
6. Chuẩn hóa đối xứng và trọng số siêu cạnh học được.
7. LayerNorm theo scope trước khi nối, nếu chuẩn embedding giữa C/P (thưa) và G (dày) lệch rõ khi theo dõi.

---

## 6. Việc cần xác minh trước khi viết kết quả

1. **`adj` rỗng?** In `sum(len(v) for v in item_adj.values())` (và entity, word). Nếu bằng 0, $k$-hop expansion của baseline không hoạt động và A0 không phản ánh thiết kế bài báo; chạy thêm một bản đã sửa làm đối chứng.
2. **Nguồn `related_*`.** Xác nhận biến thể dataset (ReDial hay HReDial) cấp node từ hội thoại hiện tại hay lịch sử user. Nếu P ≡ hội thoại hiện tại thì phần "khoảng trống C" ở §0.3 phải diễn đạt lại.
3. **Không gian chỉ số word.** `word_embedding` khai báo kích thước `n_entity` nhưng được index bằng id từ `token2id`; kiểm tra có lệch không. Điều này đặc biệt quan trọng với G vì word và entity nay cùng một đồ thị, cần offset chỉ số đúng.
4. **Thống kê G.** Phân bố kích thước siêu cạnh theo loại ($h_d$, $h^{\text{rev}}$, $h^{\text{asp}}$) và tỉ lệ node theo loại trong $\mathcal Q^G$ (để đánh giá nguy cơ word chi phối readout).
5. **$P_r$** có hay không trong codebase (§3.3).

---

## 7. Đồng bộ sang đặc tả `cpc_hypergraph_v2_1_method_spec.md`

Spec chưa được cập nhật. Các mục cần sửa:

- §2.3 và §1.1/1.4: G thành một hypergraph thống nhất trên $\mathcal V^G$, một incidence $\mathbf N^G$ (Eq. 5–8 mới); bỏ $\mathbf N^G_I,\mathbf N^G_E,\mathbf N^G_W$.
- §3.1: thêm Eq. (9G) và quy ước $\mathbf W_G:=\mathbf W_E$.
- §3.2: Eq. (9–10) thay bằng Eq. (10, 10′) mới (6 vector theo trường cho C/P, 1 vector cho G).
- §3.3 và §3.4: thay Eq. (11–12) và phần cold-start bằng §3.3 ở trên; xóa $\mathbf a_f^{(0)}$.
- §3.5: Eq. (13) thay bằng Eq. (13) mới; (14a) thành embedding sau R-GCN; $\mathbf X^G_I$ thành $\mathbf X^G[\mathcal V_I]$; $P_r$ có điều kiện.
- §4.1 và §4.2: bước dựng G chỉ xuất một $\hat{\mathbf A}^G$; forward thay các vòng `p^s_f` và `softmax(a_f)` bằng pooling 7 hàng → MHA.
- §5: bỏ dòng $\mathbf a_f^{(0)}$; thêm số head MHA (4), $L\in\{1,2\}$, giới hạn số word trong $h_d$.
- §6.2 và §7: thay bảng ablation bằng §4 ở trên; checklist bước 8 thành "cài MHA fusion trên 7 hàng", bước 1 thêm các kiểm tra ở §6.
