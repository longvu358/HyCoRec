# CPC-Hypergraph v2 — Tổng quan phương pháp (bản khớp với đặc tả)

Tài liệu này là bản tóm tắt khái niệm, giữ đồng bộ 1-1 với đặc tả kỹ thuật (`cpc_hypergraph_v2_method_spec.md`). Dùng để viết phần Method của bài báo.

---

## 0. Bức tranh từ ba nguồn gốc

### 0.1 MHIM và HyCoRec — xác nhận scope và khoảng trống

MHIM định nghĩa tường minh: *historical dialogue sessions* = các hội thoại **trước đó của cùng user**; HyCoRec kế thừa nguyên pipeline này. Kết luận về scope:

| Thành phần | MHIM | HyCoRec | Scope trong CPC |
|---|---|---|---|
| Session / Item-hypergraph | mỗi phiên lịch sử = 1 siêu cạnh trên item | giống | **P** |
| Entity-hypergraph | item lịch sử + $N$-hop DBpedia = 1 siêu cạnh | giống, $k$-hop | **P** |
| Word-hypergraph | không có | item lịch sử + $k$-hop ConceptNet | **P** |
| Hyperedge extension | truy hồi phiên user khác theo item chung | không nêu | **G dạng truy hồi** — giảm R@10 trên ReDial (nhiễu) |
| Reviews | không có | Transformer → $P_r$ | **không có hypergraph** |
| Hội thoại hiện tại | R-GCN → $N_C$, làm query vào $[N_S;N_K]$ | R-GCN → $P_c$, chỉ pooling | **không có hypergraph** |

Ba khoảng trống: (1) hội thoại hiện tại chưa có hypergraph → scope C; (2) không có cấu trúc corpus-level tĩnh → scope G; (3) review chưa được đưa vào hypergraph.

### 0.2 HyFairCRS (Zheng et al., Findings ACL 2025) — review-hypergraph đã có nhưng khác

HyFairCRS xây review-guided hypergraph: với mỗi item trong phiên lịch sử, tạo hai siêu cạnh $\{i\}\cup W^+(\mathcal{R}_i)$ và $\{i\}\cup W^-(\mathcal{R}_i)$ với $W^\pm$ là **word** có cực tính dương/âm. Cùng mẫu item-centric như entity/word, kèm line graph + contrastive.

Điểm chưa làm sau HyFairCRS:
- Node là **word**: CPC dùng **entity KG** (đạo diễn, thể loại, diễn viên…) → cùng không gian với entity-hypergraph, dùng chung $\mathbf{W}_E$.
- Phạm vi là item của phiên: CPC thêm **toàn bộ $\mathcal{V}_I$** tĩnh ở scope G → item đuôi dài có review nhưng chưa từng được nhắc vẫn nhận tín hiệu.
- Chiều item → từ: CPC thêm **aspect-centric** (entity → tập item) ở trường $I$ → siêu cạnh item không phụ thuộc hội thoại.

### 0.3 Đóng góp của CPC so với baseline

| | HyCoRec | MHIM-ext | HyFairCRS | **CPC (bản này)** |
|---|---|---|---|---|
| Hypergraph ngữ cảnh hiện tại | ✗ | ✗ | ✗ | **✓ scope C** |
| Corpus-level tĩnh | ✗ | truy hồi | ✗ | **✓ scope G** |
| Review-hypergraph, node entity KG | ✗ | ✗ | ✗ (word) | **✓ scope P + G** |
| Item đuôi dài nhận tín hiệu review | ✗ | ✗ | ✗ | **✓ scope G** |
| Aspect-centric (entity → item) | ✗ | ✗ | ✗ | **✓ scope G, trường I** |

---

## 1. Ký hiệu

- $\mathcal{V}_I\subseteq\mathcal{V}_E$, $\mathcal{V}_W$: tập item, entity DBpedia, word ConceptNet.
- Hội thoại $d$, lượt $t_s$; $\mathcal{V}_f(t_s)$ node trường $f$ tại lượt $s$; $\mathcal{V}_f(d_{\le t})=\bigcup_{s\le t}\mathcal{V}_f(t_s)$.
- $\mathcal{D}_u(d)$: tập phiên trước của user $u$ (đúng nghĩa MHIM, cắt $K_{\text{hist}}$ gần nhất); $\mathcal{I}^P=\bigcup_{d'\in\mathcal{D}_u(d)}\mathcal{V}_I(d')$.
- $\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)$: top-$k_{\text{rev}}$ entity (không phải item) liên kết từ câu review **dương** của item $i$, sắp theo tần suất.
- $\text{tf}^{+}(c,i)$: số câu review dương của $i$ chứa entity $c$; $\text{df}(c)=|\{i: c\in\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)\}|$.
- Incidence nhị phân; trọng số siêu cạnh $=1$.

---

## 2. Ba scope — dựng hypergraph

### 2.1 Scope Personal $\mathcal{H}^P$ — mở rộng HyCoRec với review

Ba hypergraph item/entity/word giữ nguyên HyCoRec:

$$
\mathcal{H}^P_I=\{h_{d'}=\mathcal{V}_I(d'):d'\in\mathcal{D}_u(d)\},\quad
\mathcal{H}^P_W=\{h_i=\{i\}\cup\mathcal{N}^{(k)}_{\text{ConceptNet}}(i):i\in\mathcal{I}^P\}. \tag{1}
$$

**[mới] Review-hypergraph scope P**, bổ sung vào trường $E$:

$$
\mathcal{H}^{P,\text{rev}}_E=\{h^{\text{rev}}_i=\{i\}\cup\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i):i\in\mathcal{I}^P\}. \tag{2}
$$

Incidence entity của scope P được ghép:

$$
\mathbf{N}^P_E = \bigl[\,\underbrace{\mathbf{N}^{P,\text{KG}}_E}_{\text{HyCoRec gốc}}\;\big|\;\underbrace{\mathbf{N}^{P,\text{rev}}_E}_{\text{mới, Eq.2}}\,\bigr]. \tag{3}
$$

Node của cả hai vế đều thuộc $\mathcal{V}_E$, nên dùng **chung** $\mathbf{X}^{(0)}_E$ và $\mathbf{W}_E$. Một lần HConv lan truyền đồng thời tín hiệu KG-expansion và tín hiệu review. Tắt hyperedge extension của MHIM nếu code kế thừa.

**Tại sao dùng entity thay vì word như HyFairCRS:** entity KG nằm cùng không gian với entity-hypergraph KG-expansion, tránh thêm embedding table; đồng thời entity (đạo diễn X, thể loại Y) mang ngữ nghĩa cụ thể hơn word thông thường.

### 2.2 Scope Contextual $\mathcal{H}^C$ — quan hệ giữa các lượt gần nhau

Mục tiêu: học quan hệ *xuyên lượt ngắn hạn* — node ở lượt $t$ và node ở lượt $t-1$, $t-2$... cùng chủ đề nhưng không nhất thiết cùng lượt. Ví dụ: user nhắc "Inception" ở lượt 3, nhắc "Christopher Nolan" ở lượt 4 — hai node liên quan nhau nhưng siêu cạnh theo từng lượt đơn lẻ không nối được chúng.

**[Cách A — đang dùng] Cửa sổ trượt:** tại mỗi vị trí $t'\le t$, tạo một siêu cạnh gộp $w$ lượt liên tiếp:

$$
\mathcal{H}^C_f(d,t)=\Bigl\{h_{t'}=\bigcup_{s=\max(1,\,t'-w+1)}^{t'}\mathcal{V}_f(t_s):t'\le t,\ |h_{t'}|\ge2\Bigr\}. \tag{4}
$$

Node trong cùng một cửa sổ được đặt vào chung một siêu cạnh → sau HConv chúng trao đổi thông tin với nhau. Tập node của $\mathcal{H}^C_f$ là $\mathcal{V}_f(d_{\le t})$. Không mở rộng $k$-hop để C là tín hiệu *co-occurrence ngắn hạn thuần túy*, phân biệt với P là *KG-expansion dài hạn*. Bỏ siêu cạnh có $|h|<2$. Giá trị khởi điểm: $w=3$; quét $\{2,3,4\}$.

**Tại sao C bổ sung cho P:** P mô hình hoá sở thích *dài hạn* qua các phiên trước và KG; C nắm bắt *mạch hội thoại ngắn hạn* — những entity và item vừa được nhắc trong vài lượt gần đây có quan hệ ngữ cảnh tức thời mà lịch sử phiên không có. Hai nguồn tín hiệu khác nhau về phạm vi thời gian và cơ chế liên kết.

**[Cách B — để dành cho lần sau]** Ghép hai loại siêu cạnh: (i) siêu cạnh lượt đơn $h_s=\mathcal{V}_f(t_s)$ bắt tín hiệu co-mention trong lượt; (ii) siêu cạnh cửa sổ Eq. (4) bắt tín hiệu xuyên lượt. Cách B học đồng thời quan hệ nội-lượt và liên-lượt nhưng tăng số siêu cạnh và có thể nhiễu; cần ablation riêng.

### 2.3 Scope Collective $\mathcal{H}^G$ — corpus-level, tĩnh

Dựng một lần trên $\mathcal{D}_{\text{train}}$; không phụ thuộc mẫu tại inference.

**(a) Siêu cạnh hội thoại**, ba trường:

$$
\mathcal{H}^{G,\text{dlg}}_f=\{h_d=\mathcal{V}_f(d):d\in\mathcal{D}_{\text{train}},\ |\mathcal{V}_f(d)|\ge2\}. \tag{5}
$$

Khác MHIM extension ở hai điểm: không truy hồi theo query hiện tại, không giới hạn số lượng theo cỡ hypergraph — toàn bộ cấu trúc đồng-xuất-hiện của tập dữ liệu được đưa vào một lần.

**(b) Review-hypergraph scope G**, trường $E$, toàn bộ $\mathcal{V}_I$, tĩnh:

$$
\mathcal{H}^{G,\text{rev}}_E=\{h^{\text{rev}}_i=\{i\}\cup\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i):i\in\mathcal{V}_I\}. \tag{6}
$$

Cùng cấu trúc với Eq. (2) nhưng trên **toàn bộ $\mathcal{V}_I$**. Item chưa từng được nhắc trong hội thoại nào — điển hình là item đuôi dài — vẫn kết nối vào mạng nếu có review. Đây là điều mà Transformer review của HyCoRec không thể làm (HyCoRec chỉ encode review của item được nhắc trong hội thoại hiện tại).

**(c) Siêu cạnh aspect-centric**, trường $I$:

$$
\mathcal{C}=\{c\in\mathcal{V}_E\setminus\mathcal{V}_I:\text{df}(c)\ge m_{\min}\},\qquad
h^{\text{asp}}_c=\text{Top-}k_{\text{asp}}\{i:c\in\mathcal{E}^{+}_{\text{top}}(\mathcal{R}_i)\}\ \text{theo tf}^{+}(c,i). \tag{7}
$$

Mỗi entity aspect $c$ nối các item có review cùng nhắc $c$ → siêu cạnh trên trường **item**, không qua entity trung gian. Loại $c\in\mathcal{V}_I$ để tránh tên phim trong review tạo cạnh item–item trực tiếp. Bổ sung cho Eq. (6): Eq. (6) làm giàu embedding entity qua đường item→entity→item; Eq. (7) tạo siêu cạnh item trực tiếp cho HConv trường $I$.

Tổng hợp incidence scope G:

$$
\mathbf{N}^G_E=[\mathbf{N}^{G,\text{dlg}}_E\mid\mathbf{N}^{G,\text{rev}}_E],\qquad
\mathbf{N}^G_I=[\mathbf{N}^{G,\text{dlg}}_I\mid\mathbf{N}^{G,\text{asp}}_I],\qquad
\mathbf{N}^G_W=\mathbf{N}^{G,\text{dlg}}_W. \tag{8}
$$

Tiền tính ma trận lan truyền thưa $\hat{\mathbf{A}}^G_f=(V^G_f)^{-1}N^G_f(E^G_f)^{-1}(N^G_f)^{\top}$; lưu sparse, không bao giờ tạo dense.

---

## 3. Mô hình

### 3.1 HConv — dùng chung trọng số ba scope

$$
\mathbf{X}^{s,(l+1)}_f=(\mathbf{V}^s_f)^{-1}\mathbf{N}^s_f(\mathbf{E}^s_f)^{-1}(\mathbf{N}^s_f)^{\top}\mathbf{X}^{s,(l)}_f\mathbf{W}^{(l)}_f,\quad s\in\{C,P,G\},\quad L=2. \tag{9}
$$

$\mathbf{W}^{(l)}_f$ **dùng chung** ba scope (bắt buộc: C và P nhỏ sẽ overfit nếu có tham số riêng; ba scope phải cùng không gian để fusion tuyến tính có nghĩa). Với $s=G$ thay $\mathbf{N}^s_f$ bằng $\hat{\mathbf{A}}^G_f$ đã tiền tính; tính một lần mỗi epoch.

**Trường $E$ đặc biệt:** $\mathbf{N}^P_E$ gộp KG + review (Eq. 3); $\mathbf{N}^G_E$ gộp hội thoại + review (Eq. 8). Một lần HConv làm lan truyền đồng thời tín hiệu KG-expansion và tín hiệu review-similarity — không tốn thêm tầng.

### 3.2 Readout và pooling

$$
\mathcal{Q}^C_f=\mathcal{V}_f(d_{\le t}),\quad
\mathcal{Q}^P_f=\mathcal{V}^P_f,\quad
\mathcal{Q}^G_f=\mathcal{V}_f(d_{\le t})\cup\mathcal{V}^P_f. \tag{10}
$$

$$
\mathbf{p}^s_f=\frac{1}{|\mathcal{Q}^s_f|}\sum_{v\in\mathcal{Q}^s_f}\mathbf{X}^s_f[v]. \tag{11}
$$

Scope G không đọc "toàn bộ đồ thị" mà gather embedding đã làm giàu tại đúng node liên quan đến user — cơ chế khác hoàn toàn so với MHIM extension (họ kéo thêm siêu cạnh của user khác; ta lan truyền toàn cục rồi đọc cục bộ).

### 3.3 Fusion ba scope

$$
\mathbf{P}_f=\sum_{s\in\{C,P,G\}}\alpha_{f,s}\,\mathbf{p}^s_f,\qquad
\boldsymbol{\alpha}_f=\text{softmax}(\mathbf{a}_f),\quad \mathbf{a}_f\in\mathbb{R}^3,\quad \mathbf{a}_f^{(0)}=(-1,0,-1). \tag{12}
$$

Khởi tạo $\approx(0.21,0.58,0.21)$ — bắt đầu gần HyCoRec. Tổng 9 tham số mới. Cold-start (lịch sử rỗng): $\mathbf{p}^P_f\leftarrow\mathbf{p}^C_f$, mask gradient $a_{f,P}$.

### 3.4 Ghép vào HyCoRec

Thay $P_i,P_e,P_w$ bằng $\mathbf{P}_I,\mathbf{P}_E,\mathbf{P}_W$. Giữ nguyên $P_r$ (Transformer review) và $P_c$ (R-GCN hội thoại hiện tại):

$$
\mathbf{P}_h=[\mathbf{P}_I;\mathbf{P}_E;\mathbf{P}_W;P_r],\qquad
P_{\text{mulrec}}=\text{Pool}([\text{Pool}(\mathbf{P}_h);P_c]). \tag{13}
$$

**Quan hệ $P_r$ và review-hypergraph:** $P_r$ đọc nội dung văn bản review của item được nhắc; review-hypergraph mô hình hóa quan hệ *cấu trúc* giữa item qua entity chung. Hai cơ chế bổ sung, không redundant. Ablation A11 kiểm tra điều này.

Điểm gợi ý: $P_{\text{rec}}=\text{softmax}(P_{\text{mulrec}}\cdot\mathbf{E}_I^{\top})$.

**Hai phương án $\mathbf{E}_I$:**
- **(a) cơ sở:** $\mathbf{E}_I=\mathbf{X}^{(0)}_I$ như HyCoRec.
- **(b) bắt buộc so sánh:** $\mathbf{E}_I=\mathbf{X}^{(0)}_I+\mathbf{X}^G_I$ — item ứng viên nhận tín hiệu scope G. Cơ chế duy nhất để $h^{\text{rev},G}$ và $h^{\text{asp}}$ tác động lên item đuôi dài ở phía ứng viên. $\mathbf{X}^G_I$ đã tính trong (9), không tốn thêm.

---

## 4. Ablation bắt buộc

Ký hiệu: **P** = Personal như HyCoRec (không có review-hypergraph); **P+R** = P thêm Eq. (2,3); **G0** = G chỉ $h_d$; **G** = G đầy đủ (Eq. 8).

| # | Cấu hình | $\mathbf{E}_I$ | Trả lời câu hỏi |
|---|---|---|---|
| A0 | P | (a) | tái hiện HyCoRec — phải khớp trước khi làm gì |
| A1 | P + C ($w=3$, Eq. 4) | (a) | cửa sổ trượt học quan hệ xuyên lượt có ích không |
| A2 | P + C ($w=1$) | (a) | suy biến về co-mention trong lượt đơn — đường cơ sở cho C |
| A3 | P + G0 | (a) | corpus tĩnh-toàn cục vs không có |
| A4 | P + MHIM-extension | (a) | corpus truy hồi — so trực tiếp với A3 |
| A5 | P+R | (a) | review-hypergraph scope P đơn lẻ |
| A6 | P + G0 + $h^{\text{rev},G}$ | (b) | review-hypergraph scope G, phía ứng viên |
| A7 | P + G0 + $h^{\text{asp}}$ | (b) | aspect-centric scope G, phía ứng viên |
| A8 | P+R + G | (b) | review đầy đủ ở P và G |
| A9 | C + P+R + G | (b) | **full** |
| A10 | A9, $\alpha=\tfrac13$ cố định | (b) | cần học $\alpha$ không |
| A11 | A9, tắt $P_r$ | (b) | $P_r$ và review-hypergraph redundant không |

Dòng A4 là quan trọng nhất về phản biện: reviewer hỏi "G khác extension MHIM chỗ nào" — câu trả lời phải là số.
Dòng A6 và A7 kỳ vọng tăng rõ nhất ở Coverage@k và Tail-Recall@10 (item đuôi dài).
Dòng A11 trả lời câu hỏi về $P_r$: nếu A9 − A11 không đáng kể thì Transformer review của HyCoRec đã bị review-hypergraph hấp thụ.

---

## 5. Để dành cho các lần sau

1. Decay $N(\lambda,w)$ trong C/P (6.2); decay inter-session cho $N^P$.
2. Gate theo node, điều kiện trên trạng thái hội thoại (6.4).
3. Trọng số TF-IDF/cực tính cho $h^{\text{rev}}$ và $h^{\text{asp}}$.
4. Siêu cạnh đồng xuất hiện SPPMI + k-truss trong G (6.3).
5. Cross-scope contrastive $C\leftrightarrow G$ (khác HyFairCRS: view = scope, không phải hypergraph ↔ line graph).
6. Chuẩn hóa đối xứng và trọng số siêu cạnh học được.
