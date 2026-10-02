# CPC-Hypergraph v2.2 — Đặc tả phương pháp để cài đặt

Tài liệu này là bản đặc tả đầy đủ (ký hiệu → dựng dữ liệu → mô hình → mất mát → huấn luyện → ablation). Mọi thứ không được nhắc tới trong đây thì **giữ nguyên code HyCoRec** (`zysensmile/HyCoRec`, nền CRSLab). Các mục có nhãn **[bắt buộc]** là phần phải cài; **[tùy chọn]** là biến thể để ablation. Đồng bộ với `cpc_hypergraph_v2_2_simple.md`.

## 0. Thay đổi so với v2.1

| Mục | v2.1 | v2.2 |
|---|---|---|
| Scope G | ba hypergraph theo trường I/E/W, ba $\hat{\mathbf A}^G_f$ | **một hypergraph thống nhất** trên $\mathcal V^G=\mathcal V_E\cup\mathcal V_W$, một $\mathbf N^G$, một $\hat{\mathbf A}^G$, một HConv (§2.3, §3.1) |
| Readout | mean theo scope × trường (9 vector) | C và P: mean theo trường (6 vector); G: mean một lần (1 vector); **7 hàng** (§3.2) |
| Fusion | $\boldsymbol\alpha_f=\text{softmax}(\mathbf a_f)$, 9 tham số toàn cục | **MHA với query $\mathbf P_c$ trên 7 hàng**, trọng số theo mẫu (§3.3) |
| Cold-start | mask gradient $a_{f,P}$ | hàng rỗng bị mask khỏi key (§3.4) |
| $\mathbf E_I$ cơ sở | $\mathbf X^{(0)}_I$ | embedding item sau R-GCN, đúng code (§3.5) |
| Baseline A0 | tái hiện bài báo | **tái hiện code HyCoRec** (§0.1) |
| Ablation fusion | A10 ($\alpha=\tfrac13$) | A0′, A10, A12, A13, F1–F4 (§6.2) |
| Readout $\mathcal Q^P_f$ | §3.2 ghi toàn bộ node, §4.2 ghi chỉ $\mathcal I^P$ (mâu thuẫn) | thống nhất: toàn bộ node của $\mathcal H^P_f$, như code |

### 0.1 Bài báo HyCoRec và code lệch nhau — chốt baseline theo code

Đã đọc code (chưa chạy). Các điểm lệch:

| Mục | Bài báo | Code (`hycorec.py`) | CPC theo |
|---|---|---|---|
| Fusion nhánh gợi ý | $\text{Pool}([\text{Pool}(P_h);P_c])$ | nối tập node (item, entity, word) → **MHA, query = embedding entity hội thoại** → mean → nối với entity hội thoại → mean | code (key là 7 hàng đã pool, §3.3) |
| Số lớp / số head HConv | $L=2$, multi-head + head-pooling | 1 `HypergraphConv` mỗi trường, 1 head | $L\in\{1,2\}$; A0 ở $L=1$ |
| $P_r$ (review Transformer) | có | không thấy trong `model/crs/hycorec/` | tùy chọn (§3.3) |
| Siêu cạnh item/entity/word | item: siêu cạnh phiên | cả ba: siêu cạnh = node + láng giềng `edger` | ghi rõ trong bài |
| Điểm gợi ý | $\mathbf E_I$ không rõ | `F.linear(u, entity_encoder(emb))` + bias | code |
| Pooling | không rõ | cấu hình `Mean` (ReDial); `Attn` là biến thể | Mean |

**Chưa xác minh (xem checklist §7, bước 0):** `_build_adjacent_matrix` có vẻ gán lại danh sách láng giềng thành `[]` trước khi duyệt, khiến mọi `adj` rỗng; nguồn của `related_item/entity/word` (hội thoại hiện tại hay lịch sử user) tùy biến thể dataset; `word_embedding` khai báo kích thước `n_entity` nhưng index bằng id từ `token2id`.

---

## 1. Ký hiệu và không gian chỉ số

### 1.1 Trường node

| Ký hiệu | Ý nghĩa | Nguồn trong CRSLab |
|---|---|---|
| $\mathcal{V}_E$, $n_E$ | tập entity DBpedia (task-related KG) | `entity2id` |
| $\mathcal{V}_I\subseteq\mathcal{V}_E$, $n_I$ | tập item (movie) | `item_ids` — item cũng là entity (MHIM) |
| $\mathcal{V}_W$, $n_W$ | tập word ConceptNet | `word2id` |
| $\mathcal{F}=\{I,E,W\}$ | ba trường (dùng cho scope C và P) | — |
| $\mathcal{V}^G=\mathcal{V}_E\cup\mathcal{V}_W$, $n_G=n_E+n_W$ | không gian node thống nhất của scope G; chỉ số entity giữ nguyên, chỉ số word cộng offset $n_E$ | `entity2id`, `word2id` |

Mỗi trường có một bảng embedding khởi tạo $\mathbf{X}^{(0)}_f\in\mathbb{R}^{n_f\times d}$, lấy đúng như HyCoRec: $\mathbf{X}^{(0)}_E$ từ R-GCN tiền huấn luyện contrastive trên DBpedia; $\mathbf{X}^{(0)}_I=\mathbf{X}^{(0)}_E[\mathcal{V}_I]$; $\mathbf{X}^{(0)}_W$ từ encoder ConceptNet của HyCoRec. Cả ba có cùng chiều $d$. Với scope G: $\mathbf{X}^{G,(0)}=[\mathbf{X}^{(0)}_E;\mathbf{X}^{(0)}_W]\in\mathbb{R}^{n_G\times d}$ (nối theo trục node).

### 1.2 Hội thoại

- Hội thoại $d=(u_d,\ (t_1,\dots,t_{T_d}),\ \text{idx}_d)$: user, dãy lượt, chỉ số thứ tự phiên của user (CRSLab/MHIM đã sắp theo thời gian).
- $\mathcal{V}_f(t_s)\subseteq\mathcal{V}_f$: node trường $f$ được CRSLab trích từ lượt $s$ (`items`, `entities`, `words` trong mỗi turn của file đã tiền xử lý).
- $\mathcal{V}_f(d_{\le t})=\bigcup_{s\le t}\mathcal{V}_f(t_s)$; $\mathcal{V}_f(d)=\mathcal{V}_f(d_{\le T_d})$.
- $\mathcal{D}_{\text{train}},\mathcal{D}_{\text{val}},\mathcal{D}_{\text{test}}$ theo phân chia của MHIM/HyCoRec (chia theo user).
- $\mathcal{D}_u(d)=\{d':u_{d'}=u_d,\ \text{idx}_{d'}<\text{idx}_d\}$: phiên **trước** $d$ của cùng user (đúng nghĩa "historical sessions" của MHIM). Cắt còn tối đa $K_{\text{hist}}$ phiên gần nhất như MHIM.

### 1.3 Reviews

- $\mathcal{R}_i=\{r_{i,1},\dots,r_{i,m_i}\}$ reviews của item $i$ (IMDb cho ReDial, Douban cho TG-ReDial, đúng nguồn HyCoRec).
- $\text{sent}(q)\in\{-1,0,+1\}$: cực tính **câu** $q$, tính bằng lexicon (VADER cho tiếng Anh; một lexicon tiếng Trung cho Douban). Quy ước: $+1$ nếu compound $\ge0.05$, $-1$ nếu $\le-0.05$, $0$ còn lại.
- $\mathcal{E}^{+}(\mathcal{R}_i)$: tập entity liên kết được (cùng entity linker của CRSLab, **bỏ** entity thuộc $\mathcal{V}_I$) từ các câu $\text{sent}=+1$ trong $\bigcup_j r_{i,j}$.
- $\text{tf}^{+}(c,i)$: số câu dương trong $\mathcal{R}_i$ chứa $c$; $\text{df}(c)=|\{i: c\in\mathcal{E}^{+}(\mathcal{R}_i)\}|$.
- $\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)=\text{Top-}k_{\text{rev}}(\mathcal{E}^{+}(\mathcal{R}_i))$ theo $\text{tf}^{+}(\cdot,i)$: tập entity nổi bật nhất từ review dương của $i$ — dùng để dựng review-hypergraph ở §2.1 và §2.3.

### 1.4 Hypergraph

Hypergraph trên trường $f$ được xác định bởi incidence nhị phân $\mathbf{N}\in\{0,1\}^{|\mathcal{V}|\times|\mathcal{H}|}$, bậc node $\mathbf{V}=\text{diag}(\mathbf{N}\mathbf{1})$, bậc siêu cạnh $\mathbf{E}=\text{diag}(\mathbf{N}^{\top}\mathbf{1})$. Node bậc 0 được giữ lại với quy ước $\mathbf{V}^{-1}_{vv}:=0$. Scope C và P dựng một hypergraph riêng cho từng trường $f$; scope G là **một** hypergraph trên $\mathcal{V}^G$ (§2.3).

Một mẫu huấn luyện là bộ $(d,t,y)$: hội thoại $d$, lượt $t$, item mục tiêu $y\in\mathcal{V}_I(t_{t+1})$ (đúng cách CRSLab sinh mẫu recommendation).

---

## 2. Dựng hypergraph — ba scope **[bắt buộc]**

Scope C và P dựng theo từng trường $\{I,E,W\}$; siêu cạnh review của scope P được ghép vào trường $E$. Scope G dựng **một** hypergraph thống nhất chứa mọi loại siêu cạnh.

### 2.1 Scope Personal $\mathcal{H}^P_f(d)$ — một phần giữ nguyên HyCoRec

Cho mẫu $(d,t)$, đặt $\mathcal{I}^P=\bigcup_{d'\in\mathcal{D}_u(d)}\mathcal{V}_I(d')$ (tập item lịch sử).

$$
\mathcal{H}^P_I=\{h_{d'}=\mathcal{V}_I(d'):d'\in\mathcal{D}_u(d)\},\quad
\mathcal{H}^P_E=\{h_i=\{i\}\cup\mathcal{N}^{(k)}_{\text{DBpedia}}(i):i\in\mathcal{I}^P\},\quad
\mathcal{H}^P_W=\{h_i=\{i\}\cup\mathcal{N}^{(k)}_{\text{ConceptNet}}(i):i\in\mathcal{I}^P\}. \tag{1}
$$

**[mới] Review-hypergraph scope P**, trường $E$ (node là entity):

$$
\mathcal{H}^{P,\text{rev}}_E=\Big\{h^{\text{rev}}_i=\{i\}\cup\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i):i\in\mathcal{I}^P,\ |\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)|\ge1\Big\}. \tag{1r}
$$

Node của $\mathcal{H}^{P,\text{rev}}_E$ là $\mathcal{I}^P\cup\bigcup_{i\in\mathcal{I}^P}\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)$ — **cùng không gian** $\mathcal{V}_E$ với $\mathcal{H}^P_E$, nên dùng chung embedding $\mathbf{X}^{(0)}_E$ và trọng số $\mathbf{W}_E$. Ghép incidence:

$$
\mathbf{N}^P_E \leftarrow [\mathbf{N}^{P,\text{KG}}_E \mid \mathbf{N}^{P,\text{rev}}_E] \tag{1s}
$$

tức nối thêm cột siêu cạnh review vào incidence entity của HyCoRec. Sau HConv, embedding node đã hấp thụ cả tín hiệu KG-expansion lẫn tín hiệu review.

Ý nghĩa: hai item lịch sử có cùng entity nổi bật trong review dương (cùng đạo diễn, cùng giải thưởng, cùng chủ đề…) được kéo gần nhau — tín hiệu **khía cạnh ẩn** mà hội thoại không nói ra. Khác HyFairCRS ở node (entity KG vs word) và ở chỗ không tách pos/neg thành siêu cạnh riêng.

Node $\mathcal{H}^P_I$ và $\mathcal{H}^P_W$ giữ nguyên Eq. (1). $k$ và $K_{\text{hist}}$ lấy theo HyCoRec/MHIM. Nếu $\mathcal{D}_u(d)=\emptyset$ thì tất cả rỗng. **Tắt** hyperedge extension của MHIM nếu code có kế thừa.

### 2.2 Scope Contextual $\mathcal{H}^C_f(d,t)$ — mới

Mục tiêu: học quan hệ *xuyên lượt ngắn hạn* trong hội thoại hiện tại — node ở lượt $t$ và node ở lượt $t-1,t-2...$ gần đây có quan hệ ngữ cảnh mà một siêu cạnh lượt đơn không nắm bắt được.

**[Cách A — đang dùng] Cửa sổ trượt:** tại mỗi vị trí $t'\le t$, tạo một siêu cạnh gộp $w$ lượt liên tiếp:

$$
\mathcal{H}^C_f(d,t)=\Bigl\{h_{t'}=\bigcup_{s=\max(1,\,t'-w+1)}^{t'}\mathcal{V}_f(t_s):t'\le t,\ |h_{t'}|\ge2\Bigr\}. \tag{2}
$$

Tập node của $\mathcal{H}^C_f$ là $\mathcal{V}_f(d_{\le t})$. Không mở rộng $k$-hop ở entity/word — C là tín hiệu *co-occurrence ngắn hạn thuần túy*, phân biệt với P là *KG-expansion*. Bỏ siêu cạnh có $|h_{t'}|<2$. Tham số $w$: khởi điểm $w=3$, quét $\{2,3,4\}$.

Khi $w=1$: thu về siêu cạnh lượt đơn (co-mention trong lượt). Đây là trường hợp suy biến dùng làm baseline A2 (xem §6.2).

**[Cách B — để dành cho lần sau]** Ghép hai loại: (i) siêu cạnh lượt đơn $h_s=\mathcal{V}_f(t_s)$ bắt co-mention trong lượt; (ii) siêu cạnh cửa sổ Eq. (2) bắt quan hệ xuyên lượt. Học đồng thời hai thang quan hệ nhưng tăng số siêu cạnh; cần ablation riêng.

### 2.3 Scope Collective $\mathcal{H}^G$ — một hypergraph thống nhất, tĩnh, dựng một lần

Node là toàn bộ $\mathcal{V}^G=\mathcal{V}_E\cup\mathcal{V}_W$ (item là tập con của $\mathcal{V}_E$). Khác C và P (mỗi trường một hypergraph), mọi loại siêu cạnh dưới đây nằm trong **cùng một** ma trận incidence.

**(a) Siêu cạnh hội thoại**, node gộp mọi loại:

$$
\mathcal{H}^{G,\text{dlg}}=\{h_d=\mathcal{V}_I(d)\cup\mathcal{V}_E(d)\cup\mathcal{V}_W(d):d\in\mathcal{D}_{\text{train}},\ |h_d|\ge2\}. \tag{3}
$$

Word có thể làm $h_d$ rất lớn; xem tham số $K^{\text{dlg}}_W$ ở §5.

**(b) Review-hypergraph**, item-centric, toàn bộ $\mathcal{V}_I$, tĩnh:

$$
\mathcal{H}^{G,\text{rev}}=\Big\{h^{\text{rev}}_i=\{i\}\cup\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i):i\in\mathcal{V}_I,\ |\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)|\ge1\Big\}. \tag{4}
$$

Cấu trúc giống Eq. (1r) nhưng trên **toàn bộ $\mathcal{V}_I$** thay vì chỉ $\mathcal{I}^P$. Item chưa từng được nhắc trong hội thoại nào vẫn vào mạng nếu có review — điều mà Transformer review của HyCoRec không làm được (HyCoRec chỉ encode review của item **được nhắc** trong hội thoại hiện tại). Sau HConv, hai item chia sẻ entity nổi bật trong review (cùng đạo diễn, thể loại, giải thưởng…) có embedding gần nhau: tín hiệu lan từ item đầu-phổ sang item đuôi-dài qua entity chung.

**(c) Siêu cạnh aspect**, chỉ chứa item — nối item theo entity chung:

$$
\mathcal{C}=\{c\in\mathcal{V}_E\setminus\mathcal{V}_I:\ \text{df}(c)\ge m_{\min}\},\qquad
h^{\text{asp}}_c=\underset{i\in\mathcal{V}_I:\,c\in\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)}{\text{Top-}k_{\text{asp}}}\ \text{tf}^{+}(c,i),\qquad
\mathcal{H}^{G,\text{asp}}=\{h^{\text{asp}}_c:c\in\mathcal{C},\ |h^{\text{asp}}_c|\ge2\}. \tag{5}
$$

Loại $c\in\mathcal{V}_I$ khỏi $\mathcal{C}$ để tên phim trong review không tạo cạnh item–item trực tiếp. Node $c$ **không** nằm trong $h^{\text{asp}}_c$ (tùy chọn: thêm $c$). Eq. (4) làm giàu embedding qua đường item→entity→item; Eq. (5) nối item trực tiếp. Hai cơ chế bổ sung nhau, nay cùng trong một đồ thị.

Tổng hợp:

$$
\mathbf{N}^G=[\mathbf{N}^{G,\text{dlg}}\mid\mathbf{N}^{G,\text{rev}}\mid\mathbf{N}^{G,\text{asp}}]\in\{0,1\}^{n_G\times|\mathcal{H}^G|}. \tag{6}
$$

Ma trận lan truyền thưa, một ma trận duy nhất:

$$
\hat{\mathbf{A}}^G=(\mathbf{V}^G)^{-1}\mathbf{N}^G(\mathbf{E}^G)^{-1}(\mathbf{N}^G)^{\top}\in\mathbb{R}^{n_G\times n_G}. \tag{7}
$$

Lưu ở dạng sparse; không bao giờ tạo dense. Chuẩn hóa $(\mathbf{E}^G)^{-1}$ làm loãng mỗi node trong siêu cạnh lớn nên **phân bố $|h|$ theo loại siêu cạnh phải được in ra** trước khi huấn luyện (§4.1 bước 8).

### 2.4 Điều kiện chống rò rỉ (kiểm tra bằng assert trong code)

1. Eq. (3) chỉ duyệt $\mathcal{D}_{\text{train}}$.
2. $\mathcal{D}_u(d)$ chỉ chứa phiên có $\text{idx}<\text{idx}_d$; với $d\in\mathcal{D}_{\text{test}}$, các phiên trước của cùng user cũng phải thuộc phần dữ liệu được phép (đúng cách MHIM dựng).
3. Eq. (4), (5): review của item $y$ (item mục tiêu) **được phép** xuất hiện trong $\mathcal{H}^{G,\text{rev}}$ và $\mathcal{H}^{G,\text{asp}}$ vì chúng tĩnh và dựng trên thông tin nội dung item, không phải nhãn hội thoại. $\text{df},\text{tf}^{+}$ không điều kiện hóa trên bất kỳ hội thoại nào.
4. Eq. (1r): $\mathcal{H}^{P,\text{rev}}_E$ dùng review của item trong $\mathcal{I}^P$ (lịch sử trước $d$). Review của item mục tiêu $y$ không được đưa vào P — assert $y\notin\mathcal{I}^P$.
5. $\mathcal{H}^C_f(d,t)$ chỉ dùng lượt $\le t$; item $y$ ở lượt $t+1$ không được xuất hiện trong bất kỳ siêu cạnh nào của C và P của mẫu đó.

---

## 3. Mô hình

### 3.1 Hypergraph convolution

Scope $s\in\{C,P\}$, trường $f\in\{I,E,W\}$, tầng $l=0..L-1$, đầu $m=1..M$:

$$
\mathbf{X}^{s,(l+1)}_{f,m}=(\mathbf{V}^s_f)^{-1}\mathbf{N}^s_f(\mathbf{E}^s_f)^{-1}(\mathbf{N}^s_f)^{\top}\mathbf{X}^{s,(l)}_f\mathbf{W}^{(l)}_{f,m},\qquad
\mathbf{X}^{s,(l+1)}_f=\frac1M\sum_{m=1}^M\mathbf{X}^{s,(l+1)}_{f,m}. \tag{8}
$$

Scope G, **một lần** cho cả đồ thị thống nhất:

$$
\mathbf{X}^{G,(l+1)}_m=\hat{\mathbf{A}}^G\,\mathbf{X}^{G,(l)}\,\mathbf{W}^{(l)}_{G,m},\qquad
\mathbf{X}^{G,(l+1)}=\frac1M\sum_{m}\mathbf{X}^{G,(l+1)}_m. \tag{8G}
$$

- Đầu vào: $\mathbf{X}^{s,(0)}_f=\mathbf{X}^{(0)}_f[\mathcal{V}^s_f]$; $\mathbf{X}^{G,(0)}=[\mathbf{X}^{(0)}_E;\mathbf{X}^{(0)}_W]$ (§1.1).
- Không phi tuyến giữa tầng. $L\in\{1,2\}$ (code dùng 1, bài báo nói 2). $M=1$ theo code; $M>1$ là tùy chọn.
- Trọng số: $\mathbf{W}_f$ **dùng chung giữa C và P** (hai scope nhỏ; tham số riêng sẽ overfit). Với G, mặc định **buộc $\mathbf{W}_G:=\mathbf{W}_E$** vì phần lớn node của G là entity/item, nên G ở cùng không gian với trường $E$ của C và P và vector $\mathbf r^G$ so sánh được với các hàng còn lại. Hàng word của G cũng đi qua $\mathbf{W}_E$; chưa rõ có tối ưu — ablation A12 tách $\mathbf{W}_G$.
- Trường $E$ của scope P đặc biệt: incidence $\mathbf{N}^P_E$ gộp KG-expansion và review (Eq. 1s). Trong G, tín hiệu review và hội thoại đã cùng một incidence (Eq. 6).
- Với $s\in\{C,P\}$ tính trên tập node cục bộ của mẫu; với G dùng $\hat{\mathbf{A}}^G$ tiền tính, tính một lần mỗi epoch.

Đầu ra: $\mathbf{X}^s_f=\mathbf{X}^{s,(L)}_f$, $\mathbf{X}^G=\mathbf{X}^{G,(L)}$.

### 3.2 Readout: bảy vector

Tập node để pooling:

$$
\mathcal{Q}^C_f=\mathcal{V}_f(d_{\le t}),\qquad
\mathcal{Q}^P_f=\mathcal{V}(\mathcal{H}^P_f)\ (\text{toàn bộ node của }\mathcal{H}^P_f,\ \text{như code}),\qquad
\mathcal{Q}^G=\bigcup_{f\in\{I,E,W\}}\bigl(\mathcal{V}_f(d_{\le t})\cup\mathcal{V}^P_f\bigr)\subseteq\mathcal{V}^G. \tag{9}
$$

$$
\mathbf{r}^s_f=\frac{1}{|\mathcal{Q}^s_f|}\sum_{v\in\mathcal{Q}^s_f}\mathbf{X}^s_f[v]\ \ (s\in\{C,P\}),\qquad
\mathbf{r}^G=\frac{1}{|\mathcal{Q}^G|}\sum_{v\in\mathcal{Q}^G}\mathbf{X}^G[v]. \tag{10}
$$

Ý nghĩa của $\mathcal{Q}^G$: scope G không đọc "toàn bộ đồ thị" mà đọc embedding **đã được làm giàu bởi toàn bộ tập dữ liệu** tại đúng những node liên quan tới user này (khác MHIM extension: họ kéo thêm *siêu cạnh* của user khác vào đồ thị cục bộ; ta lan truyền trên đồ thị toàn cục rồi đọc cục bộ).

Mean trên $\mathcal{Q}^G$ trộn node nhiều loại nên loại nhiều node nhất (thường là word) chi phối $\mathbf{r}^G$. Mặc định giữ mean đơn giản; **[tùy chọn, A13]** mean riêng từng loại (item, entity không phải item, word) rồi trung bình ba vector. Nếu $|\mathcal{Q}^G|>K_G$, cắt còn $K_G$ node, ưu tiên node thuộc $\mathcal{V}(d_{\le t})$.

### 3.3 Fusion: MHA với query là hội thoại hiện tại **[bắt buộc]**

Đúng cơ chế query-attention của `_attention_and_gating` trong HyCoRec (và Eq. 7 của MHIM), nhưng key là 7 vector đã pool:

$$
\mathbf{R}=\bigl[\,\mathbf r^C_I;\mathbf r^C_E;\mathbf r^C_W;\ \mathbf r^P_I;\mathbf r^P_E;\mathbf r^P_W;\ \mathbf r^G\,\bigr]\in\mathbb{R}^{7\times d}, \tag{11}
$$

$$
\tilde{\mathbf N}=\text{MHA}\bigl(\mathbf P_c,\ \mathbf R,\ \mathbf R\bigr)\in\mathbb{R}^{n_c\times d}, \tag{12}
$$

$$
\mathbf u=\text{Pool}\bigl([\,\text{Pool}(\tilde{\mathbf N})\,;\,\mathbf P_c\,]\bigr). \tag{13}
$$

- $\mathbf P_c\in\mathbb{R}^{n_c\times d}$: embedding R-GCN của các entity trong hội thoại hiện tại (`context_embedding` của code), $n_c=|\mathcal{V}_E(d_{\le t})|$.
- MHA là `nn.MultiheadAttention(d, 4)`; **dùng chung module và trọng số** với `item_attn` của baseline, không thêm tham số.
- Pool = mean theo hàng (cấu hình `Mean`). Với Mean: $\mathbf u=\frac{1}{n_c+1}\bigl(\text{mean}(\tilde{\mathbf N})+\sum_j\mathbf P_c[j]\bigr)$. Biến thể `Attn`: `SelfAttentionBatch` như code (ablation F4).
- Trọng số scope xuất hiện từ attention: attention mass trên hàng $(s,f)$ là $\alpha_{s,f}(d,t)$ phụ thuộc mẫu. Ghi lại để phân tích (§6.3).
- Vì mỗi hàng đã là trung bình, $\mathbf P_c$ chọn được scope/trường nhưng không chọn được node, khác code (key là từng node). Ablation F3 chạy bản node-level để đo mất mát này.

**[tùy chọn, A10] Nhãn hàng.** Các hàng chỉ khác nhau qua nội dung, nên attention không có cách tường minh để ưu tiên một scope. Thêm $\tilde{\mathbf r}_k=\mathbf r_k+\mathbf e_k$, $\mathbf e_k\in\mathbb{R}^d$, $k=1..7$, khởi tạo 0 (tổng $7d$ tham số) để mô hình bắt đầu như bản không nhãn.

**$P_r$ (có điều kiện).** Code HyCoRec công khai không dùng $P_r$. Nếu codebase có, chiếu về $d$ chiều và thêm một hàng $\mathbf r^{P_r}$ vào $\mathbf R$ (8 hàng); nếu không, bỏ ablation A11. Khi có: $P_r$ đọc nội dung văn bản review của item được nhắc; review-hypergraph học cấu trúc giữa item qua entity chung. Hai cơ chế bổ sung; nếu A11 cho thấy redundant thì là kết quả có giá trị.

### 3.4 Trường hợp thiếu hàng

Hàng không có node (chưa có lịch sử nên không có hàng P; đầu hội thoại chưa có word; $\mathcal{Q}^G=\emptyset$) **bị mask khỏi key**, không chèn hàng 0. Code gốc chèn hàng 0 cho tập rỗng nên hàng 0 vẫn nhận trọng số; nếu cần khớp số A0 tuyệt đối thì giữ hàng 0 ở A0.

| Tình huống | Xử lý |
|---|---|
| $n_c=0$ (chưa có entity hội thoại) | $\mathbf u=\text{mean}(\mathbf R)$ |
| mọi hàng của $\mathbf R$ rỗng | $\mathbf u=\text{mean}(\mathbf P_c)$ |
| cả hai rỗng | $\mathbf u=\mathbf 0$ (code gặp tình huống này tương tự) |

### 3.5 Ghép vào HyCoRec

$\mathbf u$ thay cho $P_{\text{mulrec}}$ của HyCoRec. Điểm gợi ý:

$$
P_{\text{rec}}=\text{softmax}\big(\mathbf u\cdot\mathbf E_I^{\top}+\mathbf b\big). \tag{14}
$$

**Bảng ứng viên $\mathbf E_I$:**

- (14a) **[cơ sở]** $\mathbf E_I=\mathbf X^{R\text{-GCN}}_I$ — embedding item sau R-GCN (`entity_encoder`), đúng code HyCoRec.
- (14b) **[bắt buộc chạy để so sánh]** $\mathbf E_I=\mathbf X^{R\text{-GCN}}_I+\mathbf X^{G}[\mathcal{V}_I]$ — item ứng viên nhận tín hiệu từ G. Đây là cơ chế duy nhất để $h^{\text{asp}}$ và $h^{\text{rev}}$ tác động lên item đuôi dài *ở phía ứng viên*; nếu chỉ dùng (14a) tín hiệu review chỉ qua readout và yếu. $\mathbf X^G$ đã tính trong (8G).

Nhánh sinh câu giữ nguyên HyCoRec, thay biểu diễn người dùng bằng $\mathbf u$.

### 3.6 Mất mát

$$
\mathcal{L}=\mathcal{L}_r+\mathcal{L}_c \tag{15}
$$

đúng như HyCoRec ($\mathcal{L}_r$ cross-entropy trên item, $\mathcal{L}_c$ cross-entropy token). Không thêm regularizer ở lần thử đầu. Hai nhiệm vụ huấn luyện riêng như CRSLab (rec trước, conv sau).

---

## 4. Thuật toán

### 4.1 Tiền xử lý (chạy một lần)

```
Input : CRSLab-processed ReDial/TG-ReDial (turn-level items/entities/words),
        DBpedia, ConceptNet, reviews R_i cho mọi i ∈ V_I, sentiment lexicon
Output: N^G (sparse, Eq. 6), Â^G (sparse, Eq. 7) — MỘT ma trận mỗi loại,
        E_top^+(R_i) cho mỗi i (dùng ở runtime cho scope P),
        per-user session index

# ── Bước 1: Dựng cấu trúc corpus (D_train)
1. Đọc D_train; với mỗi d: V_I(d), V_E(d), V_W(d) ← hợp các lượt
2. H^{G,dlg} ← {V_I(d) ∪ V_E(d) ∪ V_W(d) : |.| ≥ 2}     # Eq. 3
   # chỉ số word ← word_id + n_E; cắt tối đa K_W^dlg word mỗi h_d

# ── Bước 2: Xử lý review — chạy song song trên mỗi item
3. Với mỗi item i ∈ V_I:
      E_top^+(R_i) ← []
      tf_pos = defaultdict(int)
      với mỗi review r ∈ R_i, mỗi câu q ∈ sent_tokenize(r):
          nếu sent(q) = +1:
              với mỗi c ∈ link_entities(q):
                  nếu c ∉ V_I:              # loại entity là item
                      tf_pos[c] += 1
      df[c] += 1 cho c có tf_pos[c] > 0
      E_top^+(R_i) ← Top-k_rev(tf_pos)     # k_rev = |N^(k)_DBpedia|, giữ cân bằng
      # lưu ra file: {item_id: [entity_ids]}

# ── Bước 3: Siêu cạnh review — item-centric, toàn bộ V_I
4. H^{G,rev} ← {h_i = {i} ∪ E_top^+(R_i) : |E_top^+(R_i)| ≥ 1}   # Eq. 4

# ── Bước 4: Siêu cạnh aspect — chỉ chứa item
5. C ← {c : df[c] ≥ m_min, c ∉ V_I}
   với mỗi c ∈ C:
       h^asp_c ← Top-k_asp({i : c ∈ E_top^+(R_i)} theo tf^+(c,i))
       nếu |h^asp_c| ≥ 2: thêm vào H^{G,asp}
   # Eq. 5

# ── Bước 5: Lắp ráp incidence và tiền tính Â (một lần, trên V^G)
6. N^G ← column_concat(H^{G,dlg}, H^{G,rev}, H^{G,asp})   # Eq. 6, n_G × |H^G|
   V^G ← diag(N^G @ 1); E^G ← diag(N^G.T @ 1)
   Â^G ← diag(1/V^G) @ N^G @ diag(1/E^G) @ N^G.T           # Eq. 7
   lưu dạng scipy.sparse.csr

# ── Bước 6: Index phiên lịch sử
7. Với mỗi user u: D_u ← {d : user(d)=u} sắp theo idx

# ── Bước 7: Thống kê (in ra log trước khi huấn luyện)
8. Lưu:
   - |H^G| theo loại (dlg, rev, asp)
   - phân bố |h| (histogram) cho từng loại siêu cạnh; tỉ lệ word trong h_d
   - tỉ lệ node theo loại (item / entity khác / word) trong Q^G của mẫu
   - n_tail = |{i ∈ V_I : deg_dlg(i)=0, deg_rev(i)>0}|   ← luận điểm Matthew
   - n_total_item = |V_I|; tỉ lệ n_tail/n_total_item
```

### 4.2 Forward một mẫu $(d,t)$

```
# ════ Tính một lần trước khi lặp batch ════
X^G ← HConv_L(Â^G, X0_G; W_G := W_E)     # sparse matmul, n_G × d, Eq. 8G
# X^G được cache cho cả epoch nếu X0 không thay đổi trong epoch đó

# ════ Forward mỗi mẫu (d, t, y) ════
rows ← []                                  # các hàng của R, kèm nhãn k ∈ 1..7

# ── Scope C ──
for f in {I,E,W}:
    nodes_C ← V_f(d_≤t)
    nếu |nodes_C| = 0: continue
    hyperedges_C ← [∪_{s=max(1,t'-w+1)}^{t'} V_f(t_s) for t' in 1..t if |∪...| ≥ 2]   # Eq. 2
    N^C_f ← incidence(nodes_C, hyperedges_C)  # nhị phân
    X^C_f ← HConv_L(N^C_f, X0_f[nodes_C]; W_f)   # Eq. 8, W_f dùng chung với P
    rows.append(mean(X^C_f[nodes_C]))         # r^C_f, Eq. 10

# ── Scope P ──
hist ← D_u(d)[-K_hist:]
nếu hist rỗng: bỏ qua cả scope P (không thêm 3 hàng P)
ngược lại:
    I_P ← ∪_{d'∈hist} V_I(d')
    # Trường I: siêu cạnh phiên
    N^P_I ← incidence(I_P, sessions in hist)     # Eq. 1
    X^P_I ← HConv_L(N^P_I, X0_I[nodes_P_I]; W_I)
    rows.append(mean(X^P_I[nodes_P_I]))       # r^P_I, toàn bộ node của H^P_I

    # Trường E: siêu cạnh KG-expansion + review (gộp cột, Eq. 1s)
    N^P_E_KG  ← incidence_khop(I_P, DBpedia, k)
    N^P_E_rev ← incidence_rev(I_P)            # {i} ∪ E_top^+(R_i), đọc từ file tiền xử lý
    N^P_E ← [N^P_E_KG | N^P_E_rev]            # nối cột, cùng tập node E
    X^P_E ← HConv_L(N^P_E, X0_E[nodes_P_E]; W_E)
    rows.append(mean(X^P_E[nodes_P_E]))       # r^P_E, toàn bộ node

    # Trường W: giữ nguyên HyCoRec
    N^P_W ← incidence_khop(I_P, ConceptNet, k)
    X^P_W ← HConv_L(N^P_W, X0_W[nodes_P_W]; W_W)
    rows.append(mean(X^P_W[nodes_P_W]))       # r^P_W

# ── Scope G: gather từ X^G đã tính ──
Q ← ∪_f (V_f(d_≤t) ∪ nodes_P_f), ánh xạ sang chỉ số V^G (word + n_E)
nếu |Q| > K_G: cắt, ưu tiên V(d_≤t)
nếu |Q| > 0: rows.append(mean(X^G[Q]))       # r^G, Eq. 10 (A13: mean theo loại)

# ── Fusion ──
P_c ← entity_encoder(entity_embedding)[V_E(d_≤t)]    # n_c × d, R-GCN
R ← stack(rows)                               # tối đa 7 hàng; (+ e_k nếu A10)
nếu n_c = 0:      u ← mean(R)
elif |R| = 0:     u ← mean(P_c)
else:
    N~ ← MHA(P_c, R, R)                       # Eq. 12, dùng chung item_attn
    u  ← Pool([Pool(N~); P_c])                # Eq. 13

# ── Điểm số ──
E_I ← entity_encoder(entity_embedding)[V_I]   # (14a) cơ sở
      hoặc E_I + X^G[V_I]                     # (14b) item ứng viên nhận G
logits ← u · E_I^T + b
loss ← CE(logits, y)                           # Eq. 15
```

`X^G` tính **một lần mỗi epoch** (hoặc mỗi bước nếu `X0` cập nhật), sau đó gather theo `Q` của từng mẫu — không phụ thuộc mẫu nên an toàn để cache. `incidence_rev(I_P)` đọc từ dict đã tiền tính, không chạy entity linking tại runtime.

### 4.3 Độ phức tạp

Một bước: $O\big(L\,d\,[\text{nnz}(\hat{\mathbf{A}}^G)+B\sum_f(\text{nnz}(\mathbf{N}^C_f)+\text{nnz}(\mathbf{N}^P_f))]+B\,n_c\,|\mathbf R|\,d\big)$. Phần G chiếm gần hết nhưng tính một lần mỗi epoch, sau đó chỉ gather; C và P cỡ vài chục–vài trăm node mỗi mẫu; MHA chỉ trên $|\mathbf R|\le7$ hàng nên không đáng kể. Thêm $\mathbf{N}^{P,\text{rev}}_E$ vào P làm tăng $\text{nnz}(\mathbf{N}^P_E)$ thêm $k_{\text{rev}}$ mỗi item lịch sử — không đáng kể so với $k$-hop KG-expansion. Bộ nhớ thêm: một $\hat{\mathbf{A}}^G$ thưa và $\mathbf{X}^G$ ($n_G\times d$, cache theo epoch). Kích thước $\text{nnz}(\hat{\mathbf{A}}^G)$ phụ thuộc mạnh vào số word trong $h_d$; đo trước khi chạy.

---

## 5. Siêu tham số

| Ký hiệu | Giá trị khởi điểm | Ghi chú |
|---|---|---|
| $d$ | 128 | như HyCoRec (rec module) |
| $L$ | quét $\{1,2\}$ | code dùng 1; bài báo nói tối ưu tại 2 |
| $M$ | 1 | số đầu HConv (code) |
| số head MHA | 4 | như `item_attn` của HyCoRec |
| Pool | Mean | `Attn` là biến thể F4 |
| $k$ | như HyCoRec | $k$-hop cho $\mathcal{H}^P_{E/W}$ |
| $K_{\text{hist}}$ | như MHIM | số phiên lịch sử tối đa |
| $w$ | 3 | kích thước cửa sổ Eq. (2); quét $\{2,3,4\}$ |
| $k_{\text{rev}}$ | = $k$ | top entity từ review; quét {5,10,20} |
| $m_{\min}$ | 3 | aspect phải xuất hiện (dương) ở ≥3 item |
| $k_{\text{asp}}$ | 50 | chặn aspect quá chung; quét {20, 50, 100} |
| $K^{\text{dlg}}_W$ | chọn sau khi xem phân bố $|h_d|$ | số word tối đa trong $h_d$ (Eq. 3) |
| $K_G$ | chọn sau khi xem $|\mathcal{Q}^G|$ | số node tối đa trong readout G |
| $\mathbf W_G$ | $:=\mathbf W_E$ | tách là A12 |
| $\mathbf e_k$ (A10) | 0 | nhãn hàng, $7d$ tham số |
| lr, batch, optimizer | như HyCoRec | không đổi |

---

## 6. Kế hoạch thực nghiệm

### 6.1 Chỉ số
Recall/MRR/NDCG@{10,50}; Coverage@{5,10,15,20}, Iso-Index (Table 3 HyCoRec); **thêm** Tail-Recall@10 (Recall tính riêng trên item mục tiêu thuộc 80% đuôi theo tần suất train). Conversation: Dist-2/3/4.

### 6.2 Ablation bắt buộc

Ký hiệu: **P** = Personal như HyCoRec (không có review); **P+R** = P với review-hypergraph scope P (Eq. 1r, 1s); **G0** = G chỉ $h_d$ (Eq. 3); **G** = G đầy đủ ($h_d+h^{\text{rev}}+h^{\text{asp}}$, Eq. 6). Cấu hình không có scope nào thì hàng tương ứng không có trong $\mathbf R$. Mọi dòng dùng fusion §3.3 trừ A0 và các dòng F.

| # | Cấu hình | (14) | Trả lời câu hỏi |
|---|---|---|---|
| A0 | P, $L=1$, extension tắt, fusion node-level như code | (14a) | tái hiện code HyCoRec — phải khớp trước khi làm gì khác |
| A0′ | P, fusion 3 hàng (Eq. 11–13 chỉ với P) | (14a) | đổi sang fusion 7-hàng có tự đổi kết quả không |
| A1 | P + C ($w=3$, Eq. 2) | (14a) | cửa sổ trượt học quan hệ xuyên lượt có ích không |
| A2 | P + C ($w=1$) | (14a) | co-mention lượt đơn — đường cơ sở cho C |
| A3 | P + G0 | (14a) | corpus tĩnh-toàn cục, phía readout |
| A4 | P + MHIM-extension | (14a) | baseline corpus truy hồi — so với A3 |
| A5 | P+R | (14a) | review-hypergraph ở scope P (entity KG, không tách pos/neg) |
| A6 | P + G0 + $h^{\text{rev}}$ | (14b) | review-hypergraph trong G, phía ứng viên |
| A7 | P + G0 + $h^{\text{asp}}$ | (14b) | aspect trong G, phía ứng viên |
| A8 | P+R + G | (14b) | đầy đủ review ở P và G, aspect ở G |
| A9 | C + P+R + G | (14b) | full |
| A10 | A9 + nhãn hàng $\mathbf e_k$ | (14b) | attention có cần biết scope không |
| A11 | A9 thêm/bỏ $P_r$ (nếu codebase có) | (14b) | review Transformer và review-hypergraph redundant không |
| A12 | A9, $\mathbf W_G$ tách khỏi $\mathbf W_E$ | (14b) | buộc tham số G–E có hại không |
| A13 | A9, $\mathbf r^G$ cân bằng theo loại node | (14b) | word có chi phối readout G không |
| F1 | A9 không query: $\mathbf u=\text{mean}([\text{mean}(\mathbf R);\mathbf P_c])$ | (14b) | query $\mathbf P_c$ đóng góp bao nhiêu |
| F2 | A9, $\alpha$ toàn cục học trên 7 hàng (v2.1) | (14b) | trọng số theo mẫu vs toàn cục |
| F3 | A9, key là từng node (code-style) thay vì 7 hàng | (14b) | pooling trước attention mất bao nhiêu |
| F4 | A9, pooling `Attn` | (14b) | Mean vs Attn pooling |

Chuỗi A0 → A0′ → A1 tách thay đổi fusion khỏi thay đổi scope. Dòng A4 quan trọng nhất về phản biện ("G khác extension MHIM chỗ nào"): câu trả lời phải là số. A6 và A7 kỳ vọng tăng rõ nhất ở Coverage@k và Tail-Recall@10. Cả MHIM (Table 3) lẫn HyCoRec (Table 4) đều không ablation cách fusion, nên F1–F4 là kết quả mới của bài.

### 6.3 Phân tích định tính cần xuất ra
- Attention mass trên 7 hàng sau huấn luyện (thay cho $\alpha_f$ cũ), tách theo lượt $t$.
- Với A8/A9: nhóm mẫu theo $t$ (lượt), báo Recall@10 theo nhóm — kỳ vọng lợi ích của C tăng theo $t$.
- Thống kê ở bước 8 của §4.1 (item được "cứu" bởi $h^{\text{asp}}$) đặt cạnh Tail-Recall của A6 vs A5; tỉ lệ node theo loại trong $\mathcal{Q}^G$ đặt cạnh kết quả A13.

---

## 7. Checklist bàn giao cho người cài đặt

0. **Kiểm tra tiền đề trước khi chạy:**
   - In `sum(len(v) for v in item_adj.values())` (và entity, word). Nếu bằng 0, $k$-hop expansion của baseline không hoạt động; chạy thêm bản đã sửa làm đối chứng.
   - Xác nhận `related_item/entity/word` của biến thể dataset (ReDial hay HReDial) là hội thoại hiện tại hay lịch sử user.
   - Kiểm tra index của `word_embedding` (khai báo `n_entity`, index bằng `token2id`) và chốt offset $n_E$ cho $\mathcal V^G$.
   - Xác nhận codebase có $P_r$ hay không (quyết định A11).
1. Chạy A0 khớp số code HyCoRec (±0.5% tuyệt đối). Nếu không khớp, dừng.
2. Cài `build_review_index.py` (Bước 2–3 §4.1): với mỗi item $i$ → `{entity_ids: [c1,c2,...]}`. In thống kê: tỉ lệ item có $|\mathcal{E}^{+}_{\text{top}}|>0$; phân bố số entity.
3. Cài `build_collective.py` (Bước 1, 3–8 §4.1): xuất **một** $\hat{\mathbf{A}}^G$ trên $\mathcal V^G$; log `n_tail/n_total`, phân bố $|h|$ theo loại và tỉ lệ word trong $h_d$.
4. Cài `build_context(d,t)` → $\mathbf{N}^C_f$; unit test: assert $y\notin$ bất kỳ siêu cạnh nào.
5. Cài `build_personal_E(hist)` → $\mathbf{N}^{P,\text{KG}}_E$, $\mathbf{N}^{P,\text{rev}}_E$, nối cột thành $\mathbf{N}^P_E$.
6. Cài `ScopeHConv`: nhận incidence **hoặc** $\hat{\mathbf{A}}$ thưa; $\mathbf W_f$ dùng chung C và P; G dùng $\mathbf W_G:=\mathbf W_E$ (cờ để tách cho A12).
7. Cài cache `X^G`: tính sau mỗi epoch (hoặc mỗi bước nếu `X0` update), gather theo index $\mathcal Q^G$.
8. Cài fusion Eq. 11–13: pooling 7 hàng, mask hàng rỗng, MHA dùng chung `item_attn`, ghi lại attention mass; cờ cho A10, A13, F1–F4.
9. Chạy A0, A0′, A1→A13, F1→F4 trên ReDial trước; seed ×3, báo mean±std; TG-ReDial sau.
10. Xuất: attention mass theo hàng sau mỗi cấu hình; đường Recall@10 theo $t$ (lượt hội thoại); `n_tail` so với Tail-Recall của A6 vs A3.

---

## 8. Những gì cố ý chưa làm (để lần sau)

Decay $N(\lambda,w)$ trong C/P; decay giữa các phiên cho $N^P$; gate theo node điều kiện trên trạng thái hội thoại; MHA riêng từng trường rồi attention giữa các trường; LayerNorm theo scope trước khi nối (nếu chuẩn embedding giữa C/P thưa và G dày lệch rõ); trọng số TF-IDF/cực tính cho $h^{\text{rev}}$ và $h^{\text{asp}}$; siêu cạnh SPPMI + k-truss trong G; review-grounded expansion cho C; cross-scope contrastive ($C\leftrightarrow G$, khác HyFairCRS ở chỗ view = scope); chuẩn hóa đối xứng và trọng số siêu cạnh học được; G tách theo trường làm đối chứng với G thống nhất.
