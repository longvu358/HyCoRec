# CPC-Hypergraph v2 — Đặc tả phương pháp để cài đặt

Tài liệu này là bản đặc tả đầy đủ (ký hiệu → dựng dữ liệu → mô hình → mất mát → huấn luyện → ablation) cho phiên bản đơn giản. Mọi thứ không được nhắc tới trong đây thì **giữ nguyên HyCoRec** (code gốc `zysensmile/HyCoRec`, nền CRSLab). Các mục có nhãn **[bắt buộc]** là phần phải cài; **[tùy chọn]** là biến thể để ablation.

---

## 1. Ký hiệu và không gian chỉ số

### 1.1 Trường node

| Ký hiệu | Ý nghĩa | Nguồn trong CRSLab |
|---|---|---|
| $\mathcal{V}_E$, $n_E$ | tập entity DBpedia (task-related KG) | `entity2id` |
| $\mathcal{V}_I\subseteq\mathcal{V}_E$, $n_I$ | tập item (movie) | `item_ids` — item cũng là entity (MHIM) |
| $\mathcal{V}_W$, $n_W$ | tập word ConceptNet | `word2id` |
| $\mathcal{F}=\{I,E,W\}$ | ba trường | — |

Mỗi trường có một bảng embedding khởi tạo $\mathbf{X}^{(0)}_f\in\mathbb{R}^{n_f\times d}$, lấy đúng như HyCoRec: $\mathbf{X}^{(0)}_E$ từ R-GCN tiền huấn luyện contrastive trên DBpedia; $\mathbf{X}^{(0)}_I=\mathbf{X}^{(0)}_E[\mathcal{V}_I]$; $\mathbf{X}^{(0)}_W$ từ encoder ConceptNet của HyCoRec.

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

Hypergraph trên trường $f$ được xác định bởi incidence nhị phân $\mathbf{N}\in\{0,1\}^{|\mathcal{V}|\times|\mathcal{H}|}$, bậc node $\mathbf{V}=\text{diag}(\mathbf{N}\mathbf{1})$, bậc siêu cạnh $\mathbf{E}=\text{diag}(\mathbf{N}^{\top}\mathbf{1})$. Node bậc 0 được giữ lại với quy ước $\mathbf{V}^{-1}_{vv}:=0$.

Một mẫu huấn luyện là bộ $(d,t,y)$: hội thoại $d$, lượt $t$, item mục tiêu $y\in\mathcal{V}_I(t_{t+1})$ (đúng cách CRSLab sinh mẫu recommendation).

---

## 2. Dựng hypergraph — ba scope × bốn trường **[bắt buộc]**

Thêm trường review $R$ (node là entity, nguồn là review). Ba trường $\{I,E,W\}$ giữ như cũ; trường $R$ chỉ xuất hiện ở scope P và G.

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

Với mỗi trường $f$, node là $\mathcal{V}_f(d_{\le t})$ và

$$
\mathcal{H}^C_f(d,t)=\{h_s=\mathcal{V}_f(t_s):1\le s\le t,\ \mathcal{V}_f(t_s)\neq\emptyset\}. \tag{2}
$$

Không mở rộng $k$-hop ở entity/word; chỉ node được nhắc. Bỏ lượt rỗng để không sinh siêu cạnh bậc 0.

**[tùy chọn] biến thể cửa sổ:** $\mathcal{H}^{C,\text{win}}_f=\{h_{\text{win}}=\bigcup_{s=\max(1,t-w+1)}^{t}\mathcal{V}_f(t_s)\}$ (một siêu cạnh duy nhất; đây là cài đặt sliding window hiện có). Có thể dùng cả hai: $\mathcal{H}^C_f\cup\mathcal{H}^{C,\text{win}}_f$.

### 2.3 Scope Collective $\mathcal{H}^G_f$ — mới, tĩnh, dựng một lần

Node là toàn bộ $\mathcal{V}_f$.

**(a) Siêu cạnh hội thoại**, ba trường:

$$
\mathcal{H}^{G,\text{dlg}}_f=\{h_d=\mathcal{V}_f(d):d\in\mathcal{D}_{\text{train}},\ |\mathcal{V}_f(d)|\ge2\}. \tag{3}
$$

**(b) Review-hypergraph scope G**, trường $E$ — item-centric, toàn bộ $\mathcal{V}_I$, tĩnh:

$$
\mathcal{H}^{G,\text{rev}}_E=\Big\{h^{\text{rev}}_i=\{i\}\cup\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i):i\in\mathcal{V}_I,\ |\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)|\ge1\Big\}. \tag{4}
$$

Cấu trúc giống Eq. (1r) nhưng trên **toàn bộ $\mathcal{V}_I$** thay vì chỉ $\mathcal{I}^P$. Item chưa từng được nhắc trong hội thoại nào vẫn được đưa vào nếu có review — đây là điểm mà Transformer review của HyCoRec không thể làm (HyCoRec chỉ encode review của item **được nhắc** trong hội thoại hiện tại).

Sau HConv trên $\mathcal{H}^{G,\text{rev}}_E$, hai item chia sẻ entity nổi bật trong review (cùng đạo diễn, thể loại, giải thưởng…) có embedding gần nhau — cơ chế lan truyền tín hiệu từ item đầu-phổ sang item đuôi-dài qua entity chung, đóng góp trực tiếp vào Recall@10 và Coverage@k của item đuôi dài.

**(c) Siêu cạnh aspect từ review**, chỉ trường $I$ — nối item theo entity chung:

$$
\mathcal{C}=\{c\in\mathcal{V}_E\setminus\mathcal{V}_I:\ \text{df}(c)\ge m_{\min}\},\qquad
h^{\text{asp}}_c=\underset{i\in\mathcal{V}_I:\,c\in\mathcal{E}^{+}(\mathcal{R}_i)}{\text{Top-}k_{\text{asp}}}\ \text{tf}^{+}(c,i),\qquad
\mathcal{H}^{G,\text{asp}}_I=\{h^{\text{asp}}_c:c\in\mathcal{C},\ |h^{\text{asp}}_c|\ge2\}. \tag{5}
$$

Loại $c\in\mathcal{V}_I$ khỏi $\mathcal{C}$ để tên phim trong review không tạo cạnh item–item trực tiếp. Khác Eq. (4) ở trường node: Eq. (4) trên trường $E$ (item + entity), Eq. (5) trên trường $I$ (chỉ item). Hai cơ chế bổ sung: Eq. (4) làm giàu embedding entity qua message passing item→entity→item; Eq. (5) tạo siêu cạnh item trực tiếp trên trường item cho HConv trường $I$.

Tổng hợp:

$$
\mathbf{N}^G_E\leftarrow[\mathbf{N}^{G,\text{dlg}}_E\mid\mathbf{N}^{G,\text{rev}}_E],\qquad
\mathbf{N}^G_I\leftarrow[\mathbf{N}^{G,\text{dlg}}_I\mid\mathbf{N}^{G,\text{asp}}_I],\qquad
\mathbf{N}^G_W=\mathbf{N}^{G,\text{dlg}}_W. \tag{6}
$$

Từ $\mathbf{N}^G_f$ tính ma trận lan truyền thưa:

$$
\hat{\mathbf{A}}^G_f=(\mathbf{V}^G_f)^{-1}\mathbf{N}^G_f(\mathbf{E}^G_f)^{-1}(\mathbf{N}^G_f)^{\top}\in\mathbb{R}^{n_f\times n_f}. \tag{7}
$$

Lưu ở dạng sparse; không bao giờ tạo dense.

### 2.4 Điều kiện chống rò rỉ (kiểm tra bằng assert trong code)

1. Eq. (3) chỉ duyệt $\mathcal{D}_{\text{train}}$.
2. $\mathcal{D}_u(d)$ chỉ chứa phiên có $\text{idx}<\text{idx}_d$; với $d\in\mathcal{D}_{\text{test}}$, các phiên trước của cùng user cũng phải thuộc phần dữ liệu được phép (đúng cách MHIM dựng).
3. Eq. (4), (5): review của item $y$ (item mục tiêu) **được phép** xuất hiện trong $\mathcal{H}^{G,\text{rev}}_E$ và $\mathcal{H}^{G,\text{asp}}_I$ vì chúng tĩnh và dựng trên thông tin nội dung item, không phải nhãn hội thoại. $\text{df},\text{tf}^{+}$ không điều kiện hóa trên bất kỳ hội thoại nào.
4. Eq. (1r): $\mathcal{H}^{P,\text{rev}}_E$ dùng review của item trong $\mathcal{I}^P$ (lịch sử trước $d$). Review của item mục tiêu $y$ không được đưa vào P — assert $y\notin\mathcal{I}^P$.
5. $\mathcal{H}^C_f(d,t)$ chỉ dùng lượt $\le t$; item $y$ ở lượt $t+1$ không được xuất hiện trong bất kỳ siêu cạnh nào của C và P của mẫu đó.

---

## 3. Mô hình

### 3.1 Hypergraph convolution (đúng HyCoRec)

Với scope $s\in\{C,P,G\}$, trường $f\in\{I,E,W\}$, tầng $l=0..L-1$, đầu $m=1..M$:

$$
\mathbf{X}^{s,(l+1)}_{f,m}=(\mathbf{V}^s_f)^{-1}\mathbf{N}^s_f(\mathbf{E}^s_f)^{-1}(\mathbf{N}^s_f)^{\top}\mathbf{X}^{s,(l)}_f\mathbf{W}^{(l)}_{f,m},\qquad
\mathbf{X}^{s,(l+1)}_f=\frac1M\sum_{m=1}^M\mathbf{X}^{s,(l+1)}_{f,m}. \tag{8}
$$

Đầu vào $\mathbf{X}^{s,(0)}_f=\mathbf{X}^{(0)}_f[\mathcal{V}^s_f]$. Không phi tuyến giữa tầng (MHIM/HyCoRec). **Trọng số $\mathbf{W}^{(l)}_{f,m}$ dùng chung cho ba scope** — C và P nhỏ, cho tham số riêng sẽ overfit; ba scope phải cùng không gian để fusion (Eq. 11) có nghĩa.

Trường $E$ đặc biệt: incidence $\mathbf{N}^s_E$ gộp cả siêu cạnh KG-expansion lẫn siêu cạnh review (Eq. 1s cho P, Eq. 6 cho G). Một lần HConv làm lan truyền đồng thời tín hiệu cấu trúc KG và tín hiệu review — không tốn thêm tầng.

Với $s=G$, dùng $\hat{\mathbf{A}}^G_f$ đã tiền tính (Eq. 7), sparse matmul trên toàn bộ $n_f$ node, tính một lần mỗi bước gradient; với $s\in\{C,P\}$ tính trên tập node cục bộ của mẫu.

Đầu ra: $\mathbf{X}^s_f=\mathbf{X}^{s,(L)}_f$, $L=2$.

### 3.2 Readout theo scope

Tập node để pooling:

$$
\mathcal{Q}^C_f=\mathcal{V}_f(d_{\le t}),\qquad
\mathcal{Q}^P_f=\mathcal{V}^P_f\ (\text{toàn bộ node của }\mathcal{H}^P_f,\ \text{như HyCoRec}),\qquad
\mathcal{Q}^G_f=\mathcal{V}_f(d_{\le t})\cup\mathcal{V}^P_f. \tag{9}
$$

$$
\mathbf{p}^s_f=\frac{1}{|\mathcal{Q}^s_f|}\sum_{v\in\mathcal{Q}^s_f}\mathbf{X}^s_f[v]\in\mathbb{R}^d. \tag{10}
$$

Ý nghĩa của $\mathcal{Q}^G_f$: scope G không đọc "toàn bộ đồ thị" mà đọc embedding **đã được làm giàu bởi toàn bộ tập dữ liệu** tại đúng những node liên quan tới user này. Đây là chỗ khác biệt về cơ chế so với MHIM extension (họ kéo thêm *siêu cạnh* của user khác vào đồ thị cục bộ; ta lan truyền trên đồ thị toàn cục rồi đọc cục bộ).

**[tùy chọn]** thay mean pooling trong (9) bằng attention với query là readout của hội thoại hiện tại như MHIM Eq. (7). Không dùng ở lần đầu.

### 3.3 Fusion ba scope **[bắt buộc]**

$$
\boldsymbol{\alpha}_f=\text{softmax}(\mathbf{a}_f),\quad \mathbf{a}_f\in\mathbb{R}^3\ \text{học được},\quad \mathbf{a}_f^{(0)}=(-1,\,0,\,-1)\ \text{theo thứ tự }(C,P,G), \tag{11}
$$

$$
\mathbf{P}_f=\sum_{s\in\{C,P,G\}}\alpha_{f,s}\,\mathbf{p}^s_f. \tag{12}
$$

Khởi tạo (11) cho mô hình bắt đầu ở $\approx(0.21,0.58,0.21)$ — gần HyCoRec rồi học mở dần. Tổng cộng 9 tham số mới ($\mathbf{a}_I,\mathbf{a}_E,\mathbf{a}_W$, mỗi cái 3 vô hướng).

### 3.4 Trường hợp thiếu scope

- $\mathcal{D}_u(d)=\emptyset$ (user chưa có phiên trước): đặt $\mathbf{p}^P_f:=\mathbf{p}^C_f$ và **không** cập nhật $\alpha_{f,P}$ từ mẫu này (mask gradient) để tránh học lệch từ mẫu suy biến.
- $\mathcal{V}_f(d_{\le t})=\emptyset$ (đầu hội thoại chưa có node trường $f$): $\mathbf{p}^C_f:=\mathbf{0}$, $\mathbf{p}^G_f$ tính trên $\mathcal{V}^P_f$; nếu cả hai rỗng, $\mathbf{P}_f:=\mathbf{0}$ (HyCoRec cũng gặp tình huống này).

### 3.5 Ghép vào HyCoRec

Thay $P_i,P_e,P_w$ của HyCoRec bằng $\mathbf{P}_I,\mathbf{P}_E,\mathbf{P}_W$ từ Eq. (12). Giữ nguyên $P_r$ và $P_c$:

$$
\mathbf{P}_h=[\mathbf{P}_I;\mathbf{P}_E;\mathbf{P}_W;P_r],\qquad
P_{\text{mulrec}}=\text{Pool}\big([\text{Pool}(\mathbf{P}_h);P_c]\big). \tag{13}
$$

**Quan hệ giữa $P_r$ và review-hypergraph.** $P_r$ của HyCoRec encode nội dung văn bản của một vài review được chọn cho item đang được nhắc, bằng Transformer. Review-hypergraph (Eq. 1r, 4) mô hình hóa quan hệ *giữa các item* qua entity chung trong review, bằng HConv. Hai cơ chế bổ sung nhau: $P_r$ đọc nội dung, review-hypergraph học cấu trúc. Giữ cả hai ở lần đầu; nếu ablation cho thấy redundant thì là kết quả có giá trị.

Điểm gợi ý:

$$
P_{\text{rec}}=\text{softmax}\big(P_{\text{mulrec}}\cdot\mathbf{E}_I^{\top}\big). \tag{14}
$$

**Bảng ứng viên $\mathbf{E}_I$:**

- (14a) **[cơ sở]** $\mathbf{E}_I=\mathbf{X}^{(0)}_I$ như HyCoRec.
- (14b) **[bắt buộc chạy để so sánh]** $\mathbf{E}_I=\mathbf{X}^{(0)}_I+\mathbf{X}^{G}_I$ — item ứng viên nhận tín hiệu scope G. Đây là cơ chế để $h^{\text{asp}}$ và $h^{\text{rev},G}$ tác động lên item đuôi dài *ở phía ứng viên*; nếu chỉ dùng (14a) tín hiệu review chỉ qua readout và yếu. $\mathbf{X}^G_I$ đã tính trong (8), không tốn thêm.

Phần conversation task giữ nguyên HyCoRec (Eq. 16–19), chỉ thay $P_h$ bằng (13).

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
Output: N^G_I, N^G_E, N^G_W (sparse, Eq. 6), Â^G_f (sparse, Eq. 7),
        E_top^+(R_i) cho mỗi i (dùng ở runtime cho scope P),
        per-user session index

# ── Bước 1: Dựng cấu trúc corpus (D_train)
1. Đọc D_train; với mỗi d: V_I(d), V_E(d), V_W(d) ← hợp các lượt
2. H^{G,dlg}_f ← {V_f(d) : |V_f(d)| ≥ 2}   # Eq. 3

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

# ── Bước 3: Siêu cạnh review scope G — item-centric, toàn bộ V_I
4. H^{G,rev}_E ← {h_i = {i} ∪ E_top^+(R_i) : |E_top^+(R_i)| ≥ 1}   # Eq. 4

# ── Bước 4: Siêu cạnh aspect scope G — aspect-centric, trường I
5. C ← {c : df[c] ≥ m_min, c ∉ V_I}
   với mỗi c ∈ C:
       h^asp_c ← Top-k_asp({i : c ∈ E_top^+(R_i)} theo tf^+(c,i))
       nếu |h^asp_c| ≥ 2: thêm vào H^{G,asp}_I
   # Eq. 5

# ── Bước 5: Lắp ráp incidence và tiền tính Â
6. N^G_E ← stack(H^{G,dlg}_E, H^{G,rev}_E)     # sparse column concat (Eq. 6)
   N^G_I ← stack(H^{G,dlg}_I, H^{G,asp}_I)
   N^G_W ← H^{G,dlg}_W
   với mỗi f: V^G_f ← diag(N^G_f @ 1); E^G_f ← diag(N^G_f.T @ 1)
              Â^G_f ← diag(1/V^G_f) @ N^G_f @ diag(1/E^G_f) @ N^G_f.T   # Eq. 7
              lưu dạng scipy.sparse.csr

# ── Bước 6: Index phiên lịch sử
7. Với mỗi user u: D_u ← {d : user(d)=u} sắp theo idx

# ── Bước 7: Thống kê (in ra log trước khi huấn luyện)
8. Lưu:
   - m^G_f: số siêu cạnh mỗi trường
   - phân bố |h| (histogram) cho từng loại siêu cạnh
   - n_tail = |{i ∈ V_I : deg_dlg(i)=0, deg_rev(i)>0}|   ← luận điểm Matthew
   - n_total_item = |V_I|; tỉ lệ n_tail/n_total_item
```

### 4.2 Forward một mẫu $(d,t)$

```
# ════ Tính một lần trước khi lặp batch ════
for f in {I,E,W}:
    X^G_f ← HConv_L(Â^G_f, X0_f; W_f)     # sparse matmul, n_f × d, Eq. 8
    # X^G_f được cache cho cả epoch nếu X0_f không thay đổi trong epoch đó

# ════ Forward mỗi mẫu (d, t, y) ════

# ── Scope C ──
for f in {I,E,W}:
    nodes_C ← V_f(d_≤t)
    nếu |nodes_C| = 0: p^C_f ← 0; tiếp tục
    N^C_f ← incidence(nodes_C, turns ≤ t)    # Eq. 2, nhị phân
    X^C_f ← HConv_L(N^C_f, X0_f[nodes_C]; W_f)   # Eq. 8, dùng chung W_f
    p^C_f ← mean(X^C_f[nodes_C])             # Eq. 10

# ── Scope P ──
hist ← D_u(d)[-K_hist:]
I_P ← ∪_{d'∈hist} V_I(d')

# Trường I: siêu cạnh phiên
N^P_I ← incidence(I_P, sessions in hist)     # Eq. 1
X^P_I ← HConv_L(N^P_I, X0_I[I_P]; W_I)
p^P_I ← mean(X^P_I)

# Trường E: siêu cạnh KG-expansion + review (gộp cột, Eq. 1s)
nodes_E_KG ← I_P ∪ ∪_{i∈I_P} N^(k)_DBpedia(i)
N^P_E_KG ← incidence_khop(I_P, DBpedia, k)
N^P_E_rev ← incidence_rev(I_P)   # {i} ∪ E_top^+(R_i), đọc từ file tiền xử lý
N^P_E ← [N^P_E_KG | N^P_E_rev]   # nối cột, cùng tập node E
X^P_E ← HConv_L(N^P_E, X0_E[nodes_E]; W_E)   # Eq. 8
p^P_E ← mean(X^P_E[I_P])    # pooling chỉ trên item (như HyCoRec)

# Trường W: giữ nguyên HyCoRec
N^P_W ← incidence_khop(I_P, ConceptNet, k)
X^P_W ← HConv_L(N^P_W, X0_W[nodes_W]; W_W)
p^P_W ← mean(X^P_W[I_P])

# Cold-start
nếu hist rỗng: với mỗi f: p^P_f ← p^C_f; mask gradient a_f[P]

# ── Scope G: gather từ X^G đã tính ──
for f in {I,E,W}:
    Q ← V_f(d_≤t) ∪ nodes_P_f
    nếu |Q| = 0: p^G_f ← 0; tiếp tục
    p^G_f ← mean(X^G_f[Q])                   # Eq. 10

# ── Fusion ──
for f in {I,E,W}:
    P_f ← Σ_s softmax(a_f)[s] · p^s_f        # Eq. 11–12

# ── HyCoRec heads (không đổi) ──
P_h ← [P_I; P_E; P_W; P_r]
P_mulrec ← Pool([Pool(P_h); P_c])             # Eq. 13
E_I ← X0_I                                    # (14a) cơ sở
      hoặc X0_I + X^G_I                       # (14b) item ứng viên nhận G
logits ← P_mulrec · E_I^T
loss ← CE(logits, y)                           # Eq. 15
```

`X^G_f` tính **một lần mỗi epoch** (hoặc mỗi bước nếu `X0_f` cập nhật), sau đó gather theo `Q` của từng mẫu — không phụ thuộc mẫu nên an toàn để cache. `incidence_rev(I_P)` đọc từ dict đã tiền tính, không chạy entity linking tại runtime.

### 4.3 Độ phức tạp

Một bước: $O\big(L\,d\,[\sum_f\text{nnz}(\hat{\mathbf{A}}^G_f)+B\sum_f(\text{nnz}(\mathbf{N}^C_f)+\text{nnz}(\mathbf{N}^P_f))]\big)$. Phần G chiếm gần hết nhưng tính một lần mỗi epoch, sau đó chỉ gather; C và P cỡ vài chục–vài trăm node mỗi mẫu. Thêm $\mathbf{N}^{P,\text{rev}}_E$ vào P làm tăng $\text{nnz}(\mathbf{N}^P_E)$ thêm $k_{\text{rev}}$ mỗi item lịch sử — không đáng kể so với $k$-hop KG-expansion. Bộ nhớ thêm: $\hat{\mathbf{A}}^G_f$ thưa ($\text{nnz}\sim 10^5$–$10^6$ trên ReDial) và $\mathbf{X}^G_f$ ($n_f\times d$, cache theo epoch).

---

## 5. Siêu tham số

| Ký hiệu | Giá trị khởi điểm | Ghi chú |
|---|---|---|
| $d$ | 128 | như HyCoRec (rec module) |
| $L$ | 2 | HyCoRec tối ưu tại 2 |
| $M$ | như HyCoRec | số đầu HConv |
| $k$ | như HyCoRec | $k$-hop cho $\mathcal{H}^P_{E/W}$ |
| $K_{\text{hist}}$ | như MHIM | số phiên lịch sử tối đa |
| $w$ | 3 | cửa sổ, chỉ cho biến thể $\mathcal{H}^{C,\text{win}}$ |
| $k_{\text{rev}}$ | = $k$ | top entity từ review; giữ bằng $k$-hop để cân bằng tín hiệu; quét {5,10,20} |
| $m_{\min}$ | 3 | aspect phải xuất hiện (dương) ở ≥3 item |
| $k_{\text{asp}}$ | 50 | chặn aspect quá chung; quét {20, 50, 100} |
| $\mathbf{a}_f^{(0)}$ | $(-1,0,-1)$ | Eq. 10 |
| lr, batch, optimizer | như HyCoRec | không đổi |

---

## 6. Kế hoạch thực nghiệm

### 6.1 Chỉ số
Recall/MRR/NDCG@{10,50}; Coverage@{5,10,15,20}, Iso-Index (Table 3 HyCoRec); **thêm** Tail-Recall@10 (Recall tính riêng trên item mục tiêu thuộc 80% đuôi theo tần suất train). Conversation: Dist-2/3/4.

### 6.2 Ablation bắt buộc

Ký hiệu: **P** = Personal như HyCoRec (không có review); **P+R** = P với review-hypergraph scope P (Eq. 1r, 1s); **G0** = G chỉ $h_d$; **G** = G đầy đủ ($h_d+h^{\text{rev},G}+h^{\text{asp}}$).

| # | Cấu hình | (14) | Trả lời câu hỏi |
|---|---|---|---|
| A0 | P, extension tắt | (14a) | tái hiện HyCoRec — phải khớp trước khi làm gì khác |
| A1 | P + C (turn) | (14a) | hypergraph ngữ cảnh hiện tại có ích không |
| A2 | P + C (window) | (14a) | so với sliding window đang có |
| A3 | P + G0 | (14a) | corpus tĩnh-toàn cục, phía readout |
| A4 | P + MHIM-extension | (14a) | baseline corpus truy hồi — so với A3 |
| A5 | P+R | (14a) | review-hypergraph ở scope P (entity KG, không tách pos/neg) |
| A6 | P + G0 + $h^{\text{rev},G}$ | (14b) | review-hypergraph scope G, phía ứng viên |
| A7 | P + G0 + $h^{\text{asp}}$ | (14b) | aspect-centric scope G, phía ứng viên |
| A8 | P+R + G | (14b) | đầy đủ review ở P và G, aspect ở G |
| A9 | C + P+R + G | (14b) | full |
| A10 | A9 với $\alpha=\tfrac13$ cố định | (14b) | học $\alpha$ có cần không |
| A11 | A9 với $P_r$ HyCoRec bị tắt | (14b) | review Transformer và review-hypergraph redundant không |

### 6.3 Phân tích định tính cần xuất ra
- $\alpha_f$ sau huấn luyện, theo trường.
- Với A8: nhóm mẫu theo $t$ (lượt), báo Recall@10 theo nhóm — kỳ vọng lợi ích của C tăng theo $t$.
- Thống kê ở bước 7 của §4.1 (item được "cứu" bởi $h^{\text{asp}}$) đặt cạnh Tail-Recall của A6 vs A5.

---

## 7. Checklist bàn giao cho người cài đặt

1. Chạy A0 khớp số HyCoRec (±0.5% tuyệt đối). Nếu không khớp, dừng.
2. Cài `build_review_index.py` (Bước 2–3 §4.1): với mỗi item $i$ → `{entity_ids: [c1,c2,...]}`. In thống kê: tỉ lệ item có $|\mathcal{E}^{+}_{\text{top}}|>0$; phân bố số entity.
3. Cài `build_collective.py` (Bước 4–8 §4.1): xuất $\hat{\mathbf{A}}^G_f$, log thống kê `n_tail/n_total`.
4. Cài `build_context(d,t)` → $\mathbf{N}^C_f$; unit test: assert $y\notin$ bất kỳ siêu cạnh nào.
5. Cài `build_personal_E(hist)` → $\mathbf{N}^{P,\text{KG}}_E$, $\mathbf{N}^{P,\text{rev}}_E$, nối cột thành $\mathbf{N}^P_E$.
6. Cài `ScopeHConv`: nhận incidence **hoặc** $\hat{\mathbf{A}}$ thưa, **dùng chung** `W_f` qua ba scope.
7. Cài cache `X^G_f`: tính sau mỗi epoch (hoặc mỗi bước nếu `X0` update), gather theo index.
8. Cài fusion Eq. 11–12 với mask gradient §3.4.
9. Chạy A0→A11 trên ReDial trước; seed ×3, báo mean±std; TG-ReDial sau.
10. Xuất: bảng $\alpha_f$ sau mỗi cấu hình; đường Recall@10 theo $t$ (lượt hội thoại); `n_tail` so với Tail-Recall của A6 vs A3.

---

## 8. Những gì cố ý chưa làm (để lần sau)

Decay $N(\lambda,w)$ trong C/P (6.2); gate theo node điều kiện trên trạng thái hội thoại (6.4); trọng số TF-IDF/cực tính cho $h^{\text{rev}}$ và $h^{\text{asp}}$; siêu cạnh SPPMI+k-truss trong G (6.3); review-grounded expansion cho C; cross-scope contrastive ($C\leftrightarrow G$, khác HyFairCRS ở chỗ view = scope); chuẩn hóa đối xứng và trọng số siêu cạnh học được.
