# Retention ở scope C: từ thiết kế đến kết quả (HReDial)

Tài liệu này tổng hợp toàn bộ các vòng thử retention (vòng 1–7 trong `docs/retention_log.md`). Nội dung đi theo thứ tự:

1. Mục tiêu.
2. Các thiết kế đã thử.
3. Gate học được gì.
4. Kết quả valid và test, chia theo nhóm cấu hình.
5. Nhận xét.

Số liệu lấy từ log trong `log/`. Riêng vòng 1 và run `r2_uniform` / `r2_fixed_g07` seed 3407 lấy từ `retention_log.md`, vì log của chúng không có ở local.

Nếu bạn chưa quen với hướng nghiên cứu này, hãy đọc **mục 0** trước. Mục này giải thích bối cảnh, ý tưởng retention bằng một ví dụ, cách thiết kế thực nghiệm, và bản đồ các thực nghiệm đã có. Thuật ngữ được tóm tắt ở **Phụ lục**.

---

## 0. Đọc nhanh cho người mới

### 0.1 Bài toán gợi ý hội thoại và mô hình

**Conversational Recommender System (CRS)** là hệ gợi ý qua hội thoại. Người dùng (*seeker*) trò chuyện với hệ thống về sở thích, và ở mỗi lượt hệ thống phải đoán bộ phim người đó muốn. Đầu vào là toàn bộ hội thoại đến thời điểm hiện tại. Đầu ra là một bảng xếp hạng toàn bộ phim trong kho, và mô hình được chấm theo vị trí của phim đúng (*target*) trong bảng đó.

Dữ liệu chính là **HReDial**: hội thoại tiếng Anh về phim (ReDial), cộng thêm lịch sử các hội thoại trước của cùng người dùng. Bộ thứ hai là **HTGReDial** (TG-ReDial, tiếng Trung). Mọi kết quả trong tài liệu này là của HReDial.

Từ mỗi lượt hội thoại, mô hình trích ra ba loại **node** (gọi là ba *field*):

- **item**: phim được nhắc tới;
- **entity**: khái niệm trong knowledge graph DBpedia, ví dụ thể loại, đạo diễn, diễn viên;
- **word**: từ khoá trong ConceptNet.

Mô hình gốc là **HyCoRec** (ACL 2024). Hướng nghiên cứu của chúng ta (CPC-Hypergraph / ReSCOPE) tổ chức lại thông tin thành ba **scope**:

| scope | nguồn thông tin | câu hỏi nó trả lời |
|---|---|---|
| **C** (contextual) | hội thoại hiện tại | "Trong cuộc trò chuyện này người dùng đang nói về gì?" |
| **P** (personal) | các hội thoại trước của cùng người dùng | "Người này trước đây thích gì?" |
| **G** (collective) | thống kê đồng xuất hiện trên toàn bộ dữ liệu | "Những người khác nhắc các thứ này cùng nhau thì hay xem gì?" |

Mỗi scope cho ra một vector cho mỗi field. Như vậy có 7 "hàng":

- C_I, C_E, C_W: scope C, ba field item, entity, word;
- P_I, P_E, P_W: scope P, ba field;
- G: scope G.

Một lớp attention (MHA) tổng hợp 7 hàng này thành vector người dùng. Truy vấn của lớp này là các entity của hội thoại hiện tại. Cuối cùng mô hình lấy tích vô hướng giữa vector người dùng và vector của từng phim để xếp hạng. **Retention chỉ can thiệp vào scope C**, tức 3 trong 7 hàng.

### 0.2 Hypergraph và sliding window ở scope C

- **Hypergraph** là đồ thị trong đó một cạnh (*hyperedge*) có thể nối nhiều node cùng lúc, chứ không chỉ hai node.
- **Hypergraph convolution (conv, hay *propagation*)** cho các node chung một hyperedge trao đổi thông tin với nhau. Sau conv, mỗi node mang thêm thông tin của các node nằm cùng cạnh.
- **Readout (pooling)** gộp các node của một field thành một vector, chính là một hàng của scope C.

Scope C dựng hyperedge bằng **cửa sổ trượt** (*sliding window*) rộng w = 3 lượt. Mỗi cửa sổ gồm 3 lượt liên tiếp và nối mọi node được nhắc trong 3 lượt đó. Mỗi cửa sổ có một **tuổi** (*age*), tính bằng số lượt từ khi nó kết thúc đến lượt hiện tại.

Ví dụ: cần gợi ý phim ở lượt T5.

```
T1 seeker : "Mình thích phim kinh dị."                  → entity: horror
T2 system : "Bạn xem It (2017) chưa?"                    → item: It
T3 seeker : "Xem rồi, hay lắm! Giờ muốn gì đó vui hơn."  → entity: comedy
T4 system : "Superbad thì sao?"                          → item: Superbad
T5 seeker : "Xem rồi. Có phim hài lãng mạn nào không?"   → entity: romance, comedy
                                                           → target: phim hài lãng mạn

Cửa sổ (w = 3):   e1 = {T1..T3}: horror, It, comedy          tuổi 2
                  e2 = {T2..T4}: It, comedy, Superbad         tuổi 1
                  e3 = {T3..T5}: comedy, Superbad, romance    tuổi 0 (mới nhất)
```

`window_dedupe: true` gộp các cửa sổ có tập node trùng nhau, vì nếu không thì những quan hệ đó bị đếm nhiều lần.

### 0.2.1 Conv và readout cụ thể là gì

Code: `CustomHypergraphConv.forward`, `_scope_hconv`, `_pool_local`, `_pool_relations` trong `crslab/model/crs/hycorec/hycorec.py`.

Scope C được tính **riêng cho từng mẫu và từng field**:

1. Lấy các node của field đó xuất hiện trong ngữ cảnh.
2. Lấy embedding ban đầu x_v của chúng (embedding của KG / word).
3. Dựng một hypergraph nhỏ, trong đó các cửa sổ là hyperedge.
4. Chạy **conv** 2 lớp trên hypergraph đó.
5. **Readout** ra một vector.

**Conv (một lớp)** gồm 3 bước:

```
(0) x_v ← W · dropout(x_v)                          biến đổi tuyến tính (lin)
(1) node → cạnh:  m_e   = (1/|e|) Σ_{v∈e} x_v        mỗi cửa sổ = trung bình các node của nó
(2) cạnh → node:  out_v = (1/d_v) Σ_{e∋v} m_e + b     mỗi node = trung bình các cửa sổ chứa nó
```

Ở đây d_v là số cửa sổ chứa v, đếm trên incidence nhị phân. Lớp không có residual: sau conv, node v **được thay** bằng trung bình ngữ cảnh của nó. Hai lớp chạy nối tiếp nhau.

Lấy ví dụ field entity ở mục 0.2. Các node là h = horror, c = comedy, r = romance. Các cửa sổ là e1 = {h, c}, e2 = {c}, e3 = {c, r}.

```
m_e1 = (h + c)/2     m_e2 = c     m_e3 = (c + r)/2
out_h = m_e1                    = h/2 + c/2
out_c = (m_e1 + m_e2 + m_e3)/3  = h/6 + 2c/3 + r/6
out_r = m_e3                    = c/2 + r/2
```

Sau conv, horror đã "nhiễm" comedy, và comedy mang một phần horror. Đó là cái lợi của conv (thông tin cross-turn), cũng là cái hại của nó (quan hệ cũ trộn vào hiện tại).

**Gate retention chen vào bước (1)→(2).** Message m_e được nhân với α_e. Giả sử α_e1 = 0 (quên hẳn cửa sổ cũ):

| chế độ | công thức | out_h | out_c |
|---|---|---|---|
| không gate | như trên | h/2 + c/2 | h/6 + 2c/3 + r/6 |
| `scale` (v1) | out_v = (1/d_v) Σ α_e m_e | **0** (node bị xoá) | (0 + c + (c+r)/2)/3 |
| `interp` (v2) | out_v = (1/d_v) Σ [α_e m_e + (1−α_e) x_v] | **h** (giữ chính nó) | (c + c + (c+r)/2)/3 |

- Với `scale`, α nhỏ làm node teo về 0, nên loss đẩy α lên 1 (vòng 1).
- Với `interp`, α = 0 nghĩa là "không nhận message, giữ embedding của mình". Nếu α = 0 ở **mọi** cửa sổ, thì out_v = W x_v + b: scope C không còn trao đổi thông tin giữa các lượt. Đó chính là `r2_c_noprop` (`c_propagate: false`), và là điểm mà ip tự sập về (vòng 2).

**Readout** biến các node sau conv thành **một vector cho mỗi field**, tức một hàng C_I / C_E / C_W:

| kiểu | dùng ở | công thức |
|---|---|---|
| mean theo node | `r2_uniform`, `r2_fixed`, `r2_c_noprop` | r = mean_{v ∈ ngữ cảnh} out_v |
| trọng số theo node (max α) | `r2_rescope*` | w_v = max_{e∋v} α_e;  r = Σ w_v out_v / Σ w_v |
| theo relation (cửa sổ) | `r3_*` | r_e = mean_{v∈e} out_v;  β = softmax_e(relevance_e + log ρ_e);  r = Σ_e β_e r_e |

- Ở kiểu max α, mỗi node lấy α của cửa sổ *trẻ nhất* chứa nó. Phép chuẩn hoá /Σw làm chỉ còn **tỉ lệ** giữa các node là có tác dụng, nên α ≈ .01 vẫn tạo ra một trọng số recency gắt.
- Ở kiểu theo relation, đơn vị được cân là cửa sổ, chứ không phải node. Với `r3_rel_uniform`, β đều nhau, và r là trung bình các r_e. Kết quả này khác mean theo node, vì một node nằm trong nhiều cửa sổ sẽ được đếm nhiều lần. Đó là lý do `rel_uniform` không trùng `r2_uniform` (mục 4.5).

**Sau readout**, ba hàng C cùng ba hàng P và hàng G (7 hàng) đi vào MHA. Truy vấn của MHA là embedding các entity của ngữ cảnh. Kết quả là vector người dùng, dùng để chấm điểm mọi phim.

Như vậy retention có hai cách tác động lên điểm cuối:

- **ở conv**, nó đổi *nội dung* của từng node;
- **ở readout**, nó đổi *tỉ trọng* giữa các node hoặc cửa sổ trong hàng C.

### 0.3 Retention là gì, và vì sao cần nó

Câu hỏi của retention: **ở lượt T5, cửa sổ cũ e1 (horror, It) còn nên ảnh hưởng bao nhiêu?** Người dùng đã chuyển từ kinh dị sang hài, nên e1 có lẽ nên bị giảm trọng số. Nhưng nếu ở T5 họ nói "phim kinh dị hài", thì e1 lại rất quan trọng. Một cơ chế retention tốt phải nhận ra được sự khác biệt đó.

Các cách xử lý lịch sử, từ đơn giản đến phức tạp:

| cách | ý tưởng | cell tương ứng |
|---|---|---|
| uniform | mọi cửa sổ có trọng số như nhau | `r2_uniform`, `r3_rel_uniform` |
| latest-only | chỉ dùng cửa sổ mới nhất | (gần với điều `r3_evidence` tự học ra, mục 3.2) |
| decay cố định | trọng số = γ^tuổi, cũ thì nhẹ hơn theo một quy tắc cố định | `r2_fixed_g07`, `r3_rel_fixed` |
| **learned retention** | mô hình tự quyết trọng số từ nội dung | `r2_rescope*`, `r3_age`, `r3_evidence` |

**Giả thuyết RQ2**: learned retention tốt hơn ba cách còn lại, vì nó giữ được những quan hệ cũ vẫn còn đúng và bỏ được những quan hệ đã lỗi thời.

Trọng số retention có thể tác động ở hai chỗ:

1. **Ở conv**: làm yếu message đi qua cửa sổ cũ.
2. **Ở readout**: cửa sổ cũ đóng góp ít hơn vào hàng C.

Hai chỗ này cho ra các hiệu ứng rất khác nhau, và đây là một phát hiện quan trọng của tài liệu (mục 5.1).

### 0.4 Thiết kế thực nghiệm: logic chung

**(1) Mỗi lần chỉ đổi một thứ.** Mỗi *cell* là một cấu hình. Các cell được sắp thành một **thang ablation**: cell sau chỉ khác cell trước đúng một thành phần. Chênh lệch giữa hai cell kề nhau vì vậy cho biết đóng góp của thành phần đó. Ví dụ, thang `r3_*`:

```
rel_uniform → rel_fixed   : thêm decay cố định
rel_uniform → rel_attn    : thêm relevance (độ liên quan tới lượt hiện tại)
rel_attn    → age         : thêm tốc độ quên học được
age         → evidence    : tốc độ quên phụ thuộc nội dung các lượt sau
evidence    → noov / noprop : bỏ đặc trưng nhắc lại / tắt conv ở C
```

**(2) Đối chứng cho giả thuyết cạnh tranh.** Khi một biến thể tốt hơn, cần loại trừ các lời giải thích rẻ hơn:

- `r2_fixed_g07` kiểm tra: "lợi ích có phải chỉ do ưu tiên lượt gần không?"
- `r2_c_noprop` kiểm tra: "lợi ích có phải chỉ do tắt conv ở C không?"

**(3) Nhìn vào bên trong mô hình, không chỉ nhìn metric.** Sau khi train, lệnh `run_cpg.py analyze` ghi lại trọng số retention (α hoặc ρ) của từng cửa sổ trên tập test. Script `ret_alpha.py` thống kê các trọng số đó:

- chúng có giảm theo tuổi không;
- có phân biệt các cửa sổ trong cùng một mẫu không, hay chỉ là một hệ số chung của cả mẫu.

Nhờ vậy mới biết được gate đã học gì *thật sự*.

**(4) Phân tích tiền nghiệm (không train).** `evidence_probe.py` gán nhãn "hữu ích" cho mỗi cửa sổ: cửa sổ chứa target, hoặc nối với target trong đồ thị G. Sau đó nó đo xem những đặc trưng đơn giản (tuổi, số lần nhắc lại, độ liên quan với lượt hiện tại) phân biệt được cửa sổ hữu ích tới mức nào. Phân tích này cho biết **trần lợi ích** trước khi tốn GPU để train.

**(5) Quy trình train và đánh giá.**

- Chỉ train phần recommendation (`rec_only`), tối đa 10 epoch.
- Cuối mỗi epoch đánh giá trên tập **valid**. Giữ checkpoint có valid R@50 cao nhất, và dừng sớm nếu 3 epoch liên tiếp không tốt hơn.
- Checkpoint được giữ sẽ được chạy **một lần** trên tập **test**. Đó là số được báo cáo.

**(6) Seed và so sánh ghép cặp.** Seed cố định cách khởi tạo và thứ tự dữ liệu. Hai cell chạy cùng seed được so trực tiếp với nhau (Δ ghép cặp). Hiệu ứng chỉ đáng tin nếu lặp lại ở nhiều seed (3407, 42, 123). Ngoài ra, hai run chỉ được ghép cặp nếu mọi siêu tham số khác giống hệt nhau. Đó là lý do tách nhóm A và nhóm B theo `hconv_dropout` (mục 4).

**(7) Metric.** Mọi metric đều tính theo vị trí của phim đúng trong bảng xếp hạng, rồi lấy trung bình trên các mẫu.

| metric | ý nghĩa |
|---|---|
| **R@k** (Recall@k) | tỉ lệ mẫu có phim đúng nằm trong top-k. R@1 nghĩa là đoán trúng ngay vị trí đầu |
| **MRR@10** | trung bình của 1/hạng nếu phim đúng nằm trong top-10, ngược lại là 0. Nhạy với *đầu bảng* |
| **NDCG@10** | giống MRR nhưng giảm theo log của hạng, nên nhẹ tay hơn với hạng 2–10 |
| **tail R@k** | R@k chỉ tính trên các mẫu mà phim đúng thuộc **đuôi dài**: 80% phim ít xuất hiện nhất trong train. HyCoRec nhắm tới việc giảm "hiệu ứng Matthew" (chỉ gợi ý phim nổi tiếng), nên metric này quan trọng |
| rec_loss | cross-entropy trên tập valid; tăng nghĩa là đang overfit |

"**Đầu bảng**" trong tài liệu này là R@1, MRR và NDCG. "**Độ phủ**" là R@10 và R@50.

**(8) Đọc độ lớn.** Tập test có khoảng 4.8k mẫu, nên riêng nhiễu lấy mẫu đã làm R@1 dao động khoảng ±.0025. Một chênh lệch .003 ở một seed vì vậy chỉ là gợi ý, chưa phải bằng chứng (chi tiết ở mục 4.4).

### 0.5 Bản đồ các thực nghiệm đã có

| vòng | câu hỏi | cell | câu trả lời ngắn | nơi xem |
|---|---|---|---|---|
| 1 | Gate α học được (bản v1, chỉ làm yếu message) có học quên không? | `r2_rescope` | Không: α → 1, gate mở hết | 3.1 |
| 2 | Đổi gate sang nội suy, thêm prior decay thì sao? | `r2_rescope_i`, `r2_rescope_ip` | α → 0, gate tắt conv của C. R@1 tăng, R@10 giảm nhẹ | 3.1, 4.1 |
| 3 | Learned có hơn decay cố định không? | `r2_uniform`, `r2_fixed_g07` | Decay cố định ≈ uniform | 4.1 |
| 4 | (phụ) Truy vấn scope G nên dùng ngữ cảnh, lịch sử hay cả hai? | `g_q_ctx`, `g_q_hist`, `g_q_split` | Phần ngữ cảnh là phần có ích; bỏ lịch sử khỏi truy vấn không sao. Không liên quan trực tiếp tới retention | `retention_log.md` vòng 4 |
| 5 | Hiệu ứng R@1 của ip có lặp lại qua seed không? | `r2_uniform`, `r2_fixed_g07`, `r2_rescope_ip` × seed 42/123 | Có trên test (3/3 cặp), nhưng không có trên valid | 4.2 |
| 6 | (không train) Có tín hiệu nào ngoài tuổi để chọn cửa sổ không? | `evidence_probe.py` | Có: độ liên quan tới lượt hiện tại, và việc được nhắc lại. Trần thấp ở ReDial (18% mẫu có nhãn lẫn) | 1, `retention_log.md` vòng 6 |
| 7 | Cơ chế mới dựa trên bằng chứng (thang `r3_*`) có giúp không? | `r3_*`, `r2_c_noprop` | Retention không giúp. Relevance và việc tắt conv ở C giúp đầu bảng nhưng làm giảm tail | 3.2, 4.3 |

**Đang chờ:**

- chạy lại `r2_uniform` seed 3407 ở dropout 0.2;
- seed 42/123 cho `r2_c_noprop`, `r3_evidence_noprop`, `r3_rel_attn`, `r3_rel_uniform`.

Mục 6 và `docs/r3_results_hredial.md` mục 7 có lệnh chạy.

### 0.6 Quy ước đặt tên cell

- **`r2_*`**: cơ chế RetentionGate (vòng 1–5). Đơn vị là α cho mỗi cửa sổ, readout theo node.
  - `rescope` = gate học được;
  - hậu tố `_i` = nội suy (`interp`), `_ip` = nội suy + prior decay;
  - `fixed_gXX` = decay cố định γ = 0.XX;
  - `c_noprop` = tắt conv ở C.
- **`r3_*`**: cơ chế EvidenceRetention (vòng 6–7). Readout theo relation (cửa sổ).
  - `rel_*` = biến thể chỉ đổi cách readout, không có forget gate;
  - `age` / `evidence` = forget gate theo tuổi / theo bằng chứng;
  - `_noov` = bỏ đặc trưng nhắc lại, `_noprop` = tắt conv ở C.
- **Tên file**: `cpg_<dataset>_<cell>_seed<seed>.log` trong `log/<timestamp>/`. Checkpoint nằm trong `save/`. Dump trọng số nằm ở `results/cpg/dump/`.

---

## 1. Bài toán

Scope C biểu diễn ngữ cảnh hội thoại bằng các **sliding-window hyperedge**. Mỗi window e gồm các node (item, entity, word) được nhắc trong w = 3 turn liên tiếp. Window này có tuổi `age(e) = T − t_e`, tính từ turn kết thúc của nó đến turn hiện tại T. ReSCOPE (`docs/retention.md`) giữ **mọi** window từ đầu hội thoại, thay vì cắt lịch sử, và giao cho một cơ chế retention quyết định quan hệ nào còn đáng dùng. Đó là luận điểm của RQ2: *learned retention tốt hơn uniform history, latest-only và decay cố định*.

Một cơ chế retention cần trả lời ba câu hỏi thiết kế:

| câu hỏi | lựa chọn |
|---|---|
| (a) trọng số phụ thuộc vào gì? | tuổi, nội dung window, quan hệ với turn hiện tại, các turn đến sau window |
| (b) trọng số tác động ở đâu? | propagation (message qua hyperedge) và/hoặc readout (pooling thành hàng C) |
| (c) có mốc so sánh không? | trọng số độc lập từng window (σ) hay chuẩn hoá tương đối (softmax với một mốc) |

### Thống kê dữ liệu làm nền cho thiết kế

- Độ liên quan của window giảm dần theo tuổi với τ khoảng 3–4 turn, xấp xỉ γ ≈ 0.75 mỗi turn. Đây là nguồn của prior decay.
- 88% item đích ở ReDial là *cold*, tức chưa xuất hiện trong ngữ cảnh. Retention vì vậy chỉ giúp gián tiếp, qua láng giềng của item đích.
- Phân tích tiền nghiệm (`scripts/extra/evidence_probe.py`): chỉ **18%** mẫu ReDial có nhãn utility lẫn giữa các window, tức có window hữu ích và window không. Ở TG con số này là 41%. Đây là trần lợi ích của việc *chọn* window.

---

## 2. Các thiết kế đã thử

### 2.1 Đối chứng không học

| cell | công thức | tác động |
|---|---|---|
| `r2_uniform` | α_e = 1 | conv bình thường; readout là mean các node của C |
| `r2_fixed_g07` | α_e = 0.7^age | message × α ở conv, trọng số node ở readout |
| `r2_c_noprop` | α_e ≡ 0, cổng `interp` | C **không propagate**: mỗi node giữ lin(x)+b; readout là mean |

### 2.2 RetentionGate: α độc lập cho từng window (vòng 1–2)

```
α_e = σ( MLP([h_e ; h_e ⊙ q ; φ(age_e)]) + b )
```

- h_e là embedding *trước propagation* của window, nối từ 3 field.
- q là h của window mới nhất.
- φ là embedding theo bucket tuổi.
- Lớp cuối của MLP được khởi tạo bằng 0.

| phiên bản | cell | conv | khởi tạo b | readout |
|---|---|---|---|---|
| v1 `scale` | `r2_rescope` | message × α_e | σ(3) = .95 (gần uniform) | trọng số node = max α của các window chứa node, chuẩn hoá /Σw |
| v2 `interp` | `r2_rescope_i` | `out_v = D⁻¹ Σ_e [α_e m_e + (1−α_e) x_v]`; α = 0 thì node giữ chính nó | σ(3) | như v1 |
| v2 + prior | `r2_rescope_ip` | như v2 | logit(0.75^age): bắt đầu đúng ở decay thống kê | như v1 |

### 2.3 EvidenceRetention: retention theo bằng chứng (vòng 6–7)

Thiết kế lại sau khi chẩn đoán RetentionGate có lỗi cấu trúc (mục 3.1). Đơn vị bây giờ là *relation* (window). Retention chỉ được phép giảm khi có bằng chứng từ các turn đến **sau** window đó.

```
r_e^f   = mean các node của e sau conv              (conv ở C không bị gate)
f_{e,t} = σ( MLP([h_e ; u_t ; h_e ⊙ u_t ; ov(e,t)]) − 4 )   với mỗi turn t > t_e
log ρ_e = Σ_{t>t_e} log(1 − f_{e,t})                 (window mới nhất: ρ = 1, làm mốc)
β_e     = softmax_e( (W_q u_T)·(W_k h_e)/√d + log ρ_e )
c^C_f   = Σ_e β_e r_e^f
```

Trong đó ov(e,t) là tỉ lệ node của e được nhắc lại ở turn t. Mọi thành phần đều bắt đầu trung tính: W_k = 0, quên khoảng 1.8% mỗi turn.

#### 2.3.1 Giải thích từng ký hiệu

Code: `cpc.EvidenceRetention`, cùng `_evidence_retention`, `_window_reps` và `_pool_relations` trong `hycorec.py`.

Mỗi mẫu có các lượt t = 0, 1, …, T. T là lượt cuối cùng *trước* lượt cần gợi ý. Kích thước embedding là d = 128 cho mỗi field. Khi nối 3 field (item, entity, word), vector có kích thước D = 384. Field nào không có node thì phần của nó bằng 0.

| ký hiệu | là gì | tính thế nào | trực giác |
|---|---|---|---|
| e, t_e | một cửa sổ và lượt kết thúc của nó | từ `sliding_window_hyperedges` | e3 = {T3..T5} có t_e = T5 |
| **h_e** ∈ ℝ^D | "cửa sổ này nói về gì" | với mỗi field, lấy trung bình embedding **ban đầu** x⁽⁰⁾ (trước conv) của các node trong e, rồi nối 3 field | dùng x⁽⁰⁾ để điểm retention không phụ thuộc vào kết quả conv, vốn đã trộn cả ngữ cảnh |
| **u_t** ∈ ℝ^D | "lượt t nói về gì" | như h_e, nhưng trên các node được nhắc ở đúng lượt t. Lượt không nhắc gì thì u_t = 0 | |
| **q** | truy vấn "hiện tại" | u_t của lượt **không rỗng** cuối cùng | ở bảng trên viết là u_T |
| **ov(e,t)** ∈ [0,1] | mức nhắc lại | (số node của e xuất hiện lại ở lượt t, cộng qua 3 field) / (số node của e) | tín hiệu tường minh: "quan hệ này vẫn đang được dùng" |
| r_e^f ∈ ℝ^d | nội dung cửa sổ, dùng để readout | trung bình các node của e sau **conv** (mục 0.2.1), riêng từng field | đây mới là thứ được cộng vào hàng C |

Lưu ý có hai biểu diễn của một cửa sổ. **h_e** (trước conv) dùng để *quyết định trọng số*. **r_e^f** (sau conv) là *thứ được cân*.

#### 2.3.2 Forget gate: ρ_e đo "cửa sổ còn được nhớ bao nhiêu"

Với mỗi cửa sổ e và mỗi lượt t đến **sau** nó (t_e < t ≤ T), mô hình hỏi: *lượt t có làm quan hệ e lỗi thời không?*

```
z_{e,t} = MLP([h_e ; u_t ; h_e ⊙ u_t ; ov(e,t)]) − 4      MLP: Linear(3D+1 → 64) → tanh → Linear(64 → 1)
f_{e,t} = σ(z_{e,t})                                     mức quên do lượt t gây ra, trong (0,1)
ρ_e     = Π_{t > t_e} (1 − f_{e,t})                      ⇔  log ρ_e = Σ_{t>t_e} log(1 − f_{e,t})
```

Ý nghĩa từng đầu vào của MLP:

- **h_e**: quan hệ là gì. Có loại quan hệ dễ lỗi thời hơn loại khác.
- **u_t**: lượt sau nói gì.
- **h_e ⊙ u_t**: tích từng chiều, đo mức hai bên cùng chủ đề. Cùng chủ đề thì quan hệ còn đúng, chủ đề lệch hẳn thì người dùng đã chuyển hướng.
- **ov(e,t)**: lượt t có nhắc lại đúng node của e không.

Các tính chất được bảo đảm *bằng cấu trúc*, không phải nhờ học:

- **ρ chỉ có thể giảm theo thời gian**: mỗi lượt mới nhân thêm một thừa số (1 − f) ≤ 1.
- **Chỉ các lượt *sau* cửa sổ mới làm nó phai.** Cửa sổ mới nhất không có lượt nào sau nó, nên tích rỗng và ρ = 1. Đây là **mốc**: mọi cửa sổ cũ được đo *so với* cửa sổ mới nhất.
- **Hai cửa sổ cùng tuổi có thể có ρ khác nhau**, tuỳ những gì xảy ra sau chúng. RetentionGate không làm được điều này (mục 3.1).
- Lượt rỗng (u_t = 0, ov = 0) vẫn được tính là một lượt đến sau. Mức quên của nó lúc đó chỉ phụ thuộc vào h_e.

**Khởi tạo trung tính.** Lớp cuối của MLP bằng 0, nên ban đầu z = −4 và f = σ(−4) ≈ 0.018. Sau 10 lượt, ρ ≈ 0.98¹⁰ ≈ 0.83, gần với uniform. Số −4 là hằng số cố định (`base_logit`), không phải bias học được. Vì vậy weight decay kéo MLP về 0, tức kéo về trạng thái "gần như không quên". Muốn quên, mô hình phải *chủ động* học ra.

#### 2.3.3 Relevance: cửa sổ nào liên quan tới lượt hiện tại

```
s_e = (W_q q) · (W_k h_e) / √64        W_q, W_k : ℝ^384 → ℝ^64 (attention một head)
```

s_e cao nghĩa là cửa sổ e nói về điều người dùng đang nói bây giờ. W_k được khởi tạo bằng 0, nên ban đầu mọi s_e = 0. Thành phần này phải học từ đầu, và không thiên vị cửa sổ nào.

Relevance và retention trả lời hai câu hỏi khác nhau:

- **retention (ρ)**: quan hệ này *còn đúng* không, xét theo những gì xảy ra sau nó?
- **relevance (s)**: quan hệ này có *liên quan tới câu hỏi hiện tại* không?

Một cửa sổ có thể còn đúng mà không liên quan (sở thích kinh dị vẫn đúng, nhưng giờ người dùng hỏi phim hài). Nó cũng có thể đã cũ mà vẫn liên quan.

#### 2.3.4 Gộp lại: β và hàng C

```
score_e = s_e + log ρ_e
β_e     = softmax_e(score_e) = ρ_e · exp(s_e) / Σ_{e'} ρ_{e'} · exp(s_{e'})
c^C_f   = Σ_e β_e · r_e^f          (cho từng field f = item, entity, word)
```

- Cộng log ρ vào logit tương đương với **nhân** trọng số exp(s_e) với ρ_e. Retention vì vậy là một hệ số chiết khấu trên attention.
- Softmax chỉ phụ thuộc vào chênh lệch giữa các score. Vì cửa sổ mới nhất luôn có log ρ = 0, mọi ρ khác đều mang nghĩa tương đối so với nó. ρ = 0.1 nghĩa là "nặng bằng 1/10 cửa sổ mới nhất, khi relevance bằng nhau".
- score_e là **một số cho mỗi cửa sổ, dùng chung cho 3 field**. Ở mỗi field, softmax chỉ chạy trên các cửa sổ có node của field đó.
- Gradient tới ρ và s đi đường: β → hàng C → MHA → loss recommendation. Không có tín hiệu giám sát trực tiếp nào cho retention.

#### 2.3.5 Ví dụ số (ví dụ ở mục 0.2, các giá trị f và s là **giả định** để minh hoạ)

Gợi ý ở T5. Có ba cửa sổ:

- e1 = {horror, It, comedy}, kết thúc ở T3, có hai lượt đến sau là T4 và T5;
- e2 = {It, comedy, Superbad}, kết thúc ở T4, có một lượt đến sau là T5;
- e3 = {comedy, Superbad, romance}, kết thúc ở T5, là cửa sổ mới nhất.

| cửa sổ | lượt sau | ov | f (giả định) | ρ | s (giả định) | ρ·e^s | β |
|---|---|---|---|---|---|---|---|
| e1 | T4 ("Superbad?") | 0 | .5 | | | | |
| | T5 ("hài lãng mạn") | 1/3 (comedy) | .2 | .5 × .8 = **.40** | 0.1 | .44 | **.10** |
| e2 | T5 | 1/3 (comedy) | .2 | **.80** | 0.3 | 1.08 | **.26** |
| e3 | — | — | — | **1** (mốc) | 1.0 | 2.72 | **.64** |

Ở T4 hội thoại chuyển hướng và không nhắc lại gì của e1, nên e1 bị quên nhiều. Ở T5 "comedy" được nhắc lại, nên cả e1 lẫn e2 chỉ phai ít.

**Thực tế mô hình học ra** (mục 3.2): f ≈ 0.94 ở mọi lượt, bất kể nội dung. Khi đó ρ_e2 ≈ .06 và ρ_e1 ≈ .004, nên β gần như dồn hết về e3: readout chỉ còn cửa sổ mới nhất. MLP có đủ đầu vào để phân biệt các trường hợp (ov, h_e ⊙ u_t), nhưng loss không dạy nó làm điều đó.

#### 2.3.6 Các biến thể trong thang `r3_*` thay đổi chỗ nào của công thức

| cell | s_e | log ρ_e | conv ở C |
|---|---|---|---|
| `r3_rel_uniform` | 0 | 0 | có |
| `r3_rel_fixed` | 0 | (số lượt sau) · log 0.7, tức ρ = 0.7^tuổi | có |
| `r3_rel_attn` | học | 0 | có |
| `r3_age` | học | (số lượt sau) · log(1 − σ(a)), một tham số a chung cho mọi cửa sổ | có |
| `r3_evidence` | học | Σ log(1 − f_{e,t}) như trên | có |
| `r3_evidence_noov` | học | như evidence, MLP không có ov | có |
| `r3_evidence_noprop` | học | như evidence | **không**: gate `interp` = 0, out_v = W x_v + b |

Ở `rel_uniform`, β = 1/(số cửa sổ), nên readout là trung bình đều các r_e. Cách này khác mean theo node của `r2_uniform` (mục 0.2.1).

Thang ablation `r3_*`: mỗi cell bỏ một thành phần.

| cell | relevance | ρ | C propagate |
|---|---|---|---|
| `r3_rel_uniform` | ✗ | 1 | ✓ |
| `r3_rel_fixed` | ✗ | 0.7^age | ✓ |
| `r3_rel_attn` | ✓ | 1 | ✓ |
| `r3_age` | ✓ | (1−f)^age, một f học được | ✓ |
| `r3_evidence` | ✓ | forget gate theo nội dung + overlap | ✓ |
| `r3_evidence_noov` | ✓ | forget gate, không có overlap | ✓ |
| `r3_evidence_noprop` | ✓ | như evidence | ✗ |

### 2.4 So sánh các thiết kế theo ba câu hỏi

| | (a) phụ thuộc | (b) tác động | (c) mốc |
|---|---|---|---|
| fixed | chỉ tuổi | conv + readout | không |
| RetentionGate v1/v2 | nội dung + tuổi + window mới nhất | conv + readout (max α theo node) | không (σ độc lập, chuẩn hoá /Σw) |
| EvidenceRetention | relevance với u_T + nội dung các turn sau + overlap | chỉ readout (theo relation) | có (ρ = 1 ở window mới nhất, softmax) |

Lưu ý: thiết kế gốc trong `docs/retention.md` đặt readout là β = softmax(a·tanh(Wg) + log α). Cài đặt v1/v2 lại dùng trọng số node max-α chuẩn hoá /Σw. EvidenceRetention là bản đầu tiên làm đúng tinh thần của "softmax + log retention" theo relation.

---

## 3. Gate học được gì

### 3.1 RetentionGate: luôn sập về biên

Các con số dưới đây lấy từ dump trên test (`ret_alpha.py`).

| cell (dropout) | α mean | α < .05 | theo tuổi | corr(tuổi, α) | std trong mẫu / giữa mẫu |
|---|---|---|---|---|---|
| v1 `r2_rescope` | .989 | 0% (99.5% > .9) | phẳng: .9945 → .983 | −.08 | .0076 / .0117 |
| v2 `r2_rescope_i` (0.3) | .029 | 88% | — | −.13 | nhỏ hơn giữa mẫu |
| v2+prior `r2_rescope_ip` (0.3) | .011 | 96% | tuổi 0 cũng chỉ .036 | −.35 | nhỏ hơn giữa mẫu |
| v2+prior `r2_rescope_ip` (0.2) | .0094 | 97.2% | tuổi 0: .030, tuổi ≥3: ≤ .003 | −.33 | .012 / .013 |

Đọc bảng:

- **v1 mở hết** (α → 1): gate chỉ có thể *làm giảm* message, và loss không thưởng cho việc đó, nên α trôi về biên trên.
- **v2 đóng hết** (α → 0): với `interp`, α = 0 nghĩa là node giữ embedding của chính nó. Gate học cách *tắt propagation của scope C*. Prior 0.75^age bị bỏ hoàn toàn, kể cả ở tuổi 0.
- Std trong mẫu ≈ std giữa mẫu: α đóng vai *hệ số thang theo mẫu*, không phải *bộ chọn window*.
- Ở readout, trọng số max-α được chuẩn hoá /Σw. Vì vậy dù α tuyệt đối gần 0, tỉ lệ giữa các node vẫn có tác dụng: node ở window tuổi 0 nặng khoảng 10× node tuổi ≥3. Readout C của ip thực chất là **recency rất gắt**.

**Chẩn đoán cấu trúc** (vì sao không thể học quên có chọn lọc):

1. α được tính lại ở mỗi turn chỉ từ nội dung window và tuổi. Nó không biết gì về các turn đến *sau* window, như nhắc lại hay chuyển chủ đề, nên chỉ có thể là một hàm của độ gần.
2. Cùng một α vừa gate conv trên các window chồng lấn (w = 3), vừa làm trọng số readout. Nó phải gánh luôn tín hiệu "conv ở C có hại" (vì overfit), nên sập về 0.
3. Readout lấy max α theo node, nên retention thu về tuổi của node và mất cấu trúc relation.
4. σ độc lập không có mốc, nên α trôi thành thang đo.
5. Gradient tới α rất gián tiếp: C chỉ là 3 trong 7 hàng key của MHA, và target phần lớn là cold.

### 3.2 EvidenceRetention: quên gần hết

ρ của `r3_evidence` (dropout 0.2) theo tuổi:

| tuổi | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ≥8 |
|---|---|---|---|---|---|---|---|---|---|
| n window | 4772 | 4497 | 3816 | 3334 | 2442 | 1999 | 1343 | 1045 | 2107 |
| ρ mean | 1.000 | .057 | .012 | .003 | .003 | .0002 | .0008 | .0002 | 0 |

- 77% window có ρ < .05, corr(tuổi, ρ) = −.51. Ngay ở turn đến sau đầu tiên, forget gate đã quên khoảng 94%. Readout C của evidence vì vậy gần như **chỉ dùng window mới nhất**.
- `r3_age` đi theo hướng ngược lại: tốc độ quên học được gần như không rời khởi tạo (~1.8%/turn). Kết quả của nó trùng `r3_rel_attn` đến 3–4 chữ số ở mọi epoch.
- Cả ba họ cơ chế đều rơi về một điểm cực: mở hết, đóng hết, hoặc quên hết. Không cơ chế nào học được một mức retention trung gian có chọn lọc.

---

## 4. Kết quả: valid và test

### Thiết lập chung

- HReDial, rec-only, tối đa 10 epoch, AdamW lr 5e-4, wd 0.1.
- Early stop theo **valid R@50**, impatience 3. Test chạy trên checkpoint của epoch có valid R@50 cao nhất.
- `ep` trong các bảng là epoch được chọn trên tổng số epoch đã chạy, đánh số từ 0.

**Cảnh báo cấu hình.** Các run chia thành hai nhóm theo `hconv_dropout`, và **chỉ so sánh ghép cặp trong cùng một nhóm**:

- Nhóm A: dropout 0.3.
- Nhóm B: dropout 0.2.

Trong các bảng dưới đây, các ô "–" là số không có trong ghi chép.

### 4.1 Nhóm A: dropout 0.3, seed 3407 (RetentionGate)

**Valid:**

| cell | ep | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| r2_uniform | 5 | – | .1977 | .4226 | – | – |
| r2_fixed_g07 | 4 | – | .1907 | .4180 | – | – |
| r2_rescope_i | 4/8 | .0308 | .2000 | .4194 | .0727 | .1023 |
| r2_rescope_ip | 4/8 | .0319 | .2015 | .4168 | .0734 | .1031 |

**Test:**

| cell | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | tail R@10 | tail R@50 |
|---|---|---|---|---|---|---|---|
| r2_uniform | .0265 | .1692 | .3537 | .0624 | .0872 | .0057 | .0412 |
| r2_fixed_g07 | .0272 | .1688 | .3549 | .0624 | .0871 | .0067 | .0373 |
| r2_rescope_i | .0309 | .1667 | .3566 | .0653 | .0889 | .0077 | .0373 |
| r2_rescope_ip | .0307 | .1671 | .3528 | .0653 | .0891 | .0077 | .0383 |

**Δ test so với uniform:**

| cell | ΔR@1 | ΔR@10 | ΔR@50 | ΔMRR@10 | ΔNDCG@10 |
|---|---|---|---|---|---|
| fixed_g07 | +.0007 | −.0004 | +.0012 | .0000 | −.0001 |
| rescope_i | **+.0044** | −.0025 | +.0029 | **+.0029** | +.0017 |
| rescope_ip | **+.0042** | −.0021 | −.0009 | **+.0029** | +.0019 |

Vòng 1 (v1 `scale`, `r2_rescope`) không có log ở local, và dropout của nó chưa rõ. Ghi chép cho valid R@10 .1904. Test R@10 ghi không nhất quán giữa vòng 1 (.1707) và bảng vòng 2 (.1665). Test R@1 là .031.

### 4.2 Nhóm B: dropout 0.2, nhiều seed (RetentionGate so với decay cố định)

**Valid:**

| cell | seed | ep | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| r2_uniform | 42 | 4/8 | .0348 | .2017 | .4186 | .0746 | .1040 |
| r2_uniform | 123 | 3/7 | .0345 | .1907 | .4192 | .0723 | .0997 |
| r2_fixed_g07 | 42 | 4/8 | .0331 | .1983 | .4180 | .0729 | .1020 |
| r2_fixed_g07 | 123 | 3/7 | .0340 | .1919 | .4177 | .0722 | .0999 |
| r2_rescope_ip | 42 | 3/7 | .0345 | .1959 | .4154 | .0747 | .1029 |
| r2_rescope_ip | 123 | 3/7 | .0305 | .1907 | .4206 | .0701 | .0981 |
| r2_rescope_ip | 3407 | 4/8 | .0325 | .2012 | .4218 | .0747 | .1041 |

**Test:**

| cell | seed | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | tail R@10 | tail R@50 |
|---|---|---|---|---|---|---|---|---|
| r2_uniform | 42 | .0309 | .1663 | .3499 | .0647 | .0883 | .0105 | .0402 |
| r2_uniform | 123 | .0270 | .1709 | .3510 | .0617 | .0870 | .0057 | .0297 |
| r2_fixed_g07 | 42 | .0313 | .1696 | .3539 | .0651 | .0893 | .0115 | .0469 |
| r2_fixed_g07 | 123 | .0270 | .1713 | .3535 | .0621 | .0874 | .0067 | .0297 |
| r2_rescope_ip | 42 | .0334 | .1648 | .3489 | .0665 | .0894 | .0057 | .0306 |
| r2_rescope_ip | 123 | .0322 | .1659 | .3604 | .0655 | .0888 | .0086 | .0325 |
| r2_rescope_ip | 3407 | .0324 | .1700 | .3537 | .0671 | .0910 | .0077 | .0364 |

**Δ ghép cặp so với uniform, seed 42 / 123:**

| cell | split | ΔR@1 | ΔR@10 | ΔR@50 | ΔMRR@10 | ΔNDCG@10 |
|---|---|---|---|---|---|---|
| fixed_g07 | test | +.0004 / .0000 | +.0033 / +.0004 | +.0040 / +.0025 | +.0004 / +.0004 | +.0010 / +.0004 |
| fixed_g07 | valid | −.0017 / −.0005 | −.0034 / +.0012 | −.0006 / −.0015 | −.0017 / −.0001 | −.0020 / +.0002 |
| rescope_ip | test | **+.0025 / +.0052** | −.0015 / −.0050 | −.0010 / +.0094 | **+.0017 / +.0038** | +.0010 / +.0019 |
| rescope_ip | valid | −.0003 / **−.0041** | −.0058 / .0000 | −.0032 / +.0014 | .0000 / −.0022 | −.0011 / −.0016 |

#### Đường cong valid và đỉnh của từng biến thể (nhóm B, RetentionGate)

Mỗi ô ghi giá trị đỉnh của metric trên valid và epoch đạt đỉnh, theo dạng `giá trị @ep`. Cột "ep chọn" là epoch có valid R@50 cao nhất, tức checkpoint đem đi test.

| cell | seed | ep chọn / số ep | đỉnh R@1 | đỉnh R@10 | đỉnh R@50 | đỉnh MRR@10 | valid loss thấp nhất |
|---|---|---|---|---|---|---|---|
| r2_uniform | 42 | 4 / 8 | .0363 @5 | .2017 @4 | .4186 @4 | .0760 @5 | 7.097 @2 |
| r2_uniform | 123 | 3 / 7 | .0345 @3 | .1936 @4 | .4192 @3 | .0722 @3 | 7.090 @2 |
| r2_fixed_g07 | 42 | 4 / 8 | .0334 @2 | .1983 @4 | .4180 @4 | .0742 @2 | 7.096 @2 |
| r2_fixed_g07 | 123 | 3 / 7 | .0340 @3 | .1919 @3 | .4177 @3 | .0725 @2 | 7.092 @2 |
| r2_rescope_ip | 42 | 3 / 7 | .0351 @4 | .1962 @4 | .4154 @3 | .0746 @3 | 7.155 @2 |
| r2_rescope_ip | 123 | 3 / 7 | .0348 @1 | .1948 @4 | .4206 @3 | .0705 @2 | 7.086 @2 |
| r2_rescope_ip | 3407 | 4 / 8 | .0377 @6 | .2012 @4 | .4218 @4 | .0750 @6 | 7.122 @2 |

Dưới đây là cell tốt nhất trên **test** theo từng metric và từng seed. Bảng chỉ so ba cell của nhóm B ở seed 42 và 123, vì đây là những seed có cả ba cell.

| seed | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | tail R@50 |
|---|---|---|---|---|---|---|
| 42 | ip .0334 | fixed .1696 | fixed .3539 | ip .0665 | ip .0894 | fixed .0469 |
| 123 | ip .0322 | fixed .1713 | ip .3604 | ip .0655 | ip .0888 | ip .0325 |

**Nhận xét bảng 4.2:**

- **Cả ba cell có cùng một kiểu đường cong.** Valid loss thấp nhất ở epoch 2 rồi tăng đều, khoảng 7.09 → 7.68 ở epoch 7. Trong khi đó R@10/R@50 vẫn tăng đến epoch 3–4. Checkpoint luôn được chọn **sau** điểm loss thấp nhất, tức lúc mô hình đã bắt đầu overfit về xác suất nhưng thứ hạng vẫn còn tốt lên. Sau đỉnh, R@50 rơi khoảng .01–.015 trong 3 epoch rồi early stop kích hoạt.
- **`r2_uniform`**, đối chứng không retention, đạt đỉnh R@10/R@50 ở epoch 3–4. Đỉnh R@1 và MRR rơi vào cùng epoch hoặc muộn hơn một epoch (seed 42: R@1 .0363 ở epoch 5). Vì checkpoint được chọn theo R@50, đỉnh R@1 của seed 42 bị bỏ lỡ: chọn epoch 4 có R@1 .0348.
- **`r2_fixed_g07`** (α = 0.7^tuổi ở cả conv lẫn readout) **gần như chồng khít uniform.**
  - Valid loss trùng đến 3 chữ số ở mọi epoch (7.096 so với 7.097 ở seed 42), và đỉnh rơi vào cùng epoch.
  - Trên valid nó hơi thấp hơn uniform (R@10 −.0034 ở seed 42), trên test thì hơi cao hơn (R@50 +.0040 / +.0025). Nó là cell tốt nhất về R@10 trên test ở cả hai seed.
  - Đây là thay đổi nhỏ nhất trong nhóm: decay cố định chỉ dịch nhẹ độ phủ và không đụng tới đầu bảng.
  - Đỉnh R@1 và MRR của fixed đến **sớm** (epoch 2 ở seed 42) rồi đi ngang. Decay làm mô hình bão hoà đầu bảng sớm hơn.
- **`r2_rescope_ip`** (gate học được, đã sập về α ≈ 0) **có đường cong valid khác hẳn ở đầu bảng.**
  - Đỉnh R@1 trên valid thất thường: seed 123 đạt đỉnh ngay epoch 1 (.0348) rồi rơi xuống .0305 ở epoch được chọn; seed 3407 đạt đỉnh muộn ở epoch 6 (.0377).
  - Đỉnh R@50 trên valid thấp hơn hoặc ngang uniform (.4154 so với .4186 ở seed 42), và đến sớm hơn một epoch (epoch 3).
  - Trên test, ip là cell **tốt nhất về R@1, MRR và NDCG ở mọi seed**, và tệ nhất về R@10 ở seed 42/123 (.1648 / .1659).
  - Như vậy ip không "tốt hơn đều": nó **đổi R@10 lấy R@1**. Mô hình đặt phim đúng lên vị trí 1 thường xuyên hơn, nhưng đẩy một số mẫu khác ra ngoài top-10.
  - Valid loss của ip ở seed 42 cao hơn uniform (7.155 so với 7.097), cho thấy tắt conv ở C làm mô hình khớp dữ liệu kém hơn một chút về xác suất. Dù vậy, thứ tự đầu bảng vẫn tốt hơn.
- **Đỉnh nằm ở đâu.**
  - Trên test, đầu bảng (R@1, MRR, NDCG) có đỉnh ở **ip**, còn độ phủ (R@10) có đỉnh ở **fixed**. R@50 thì chia đôi: fixed ở seed 42, ip ở seed 123.
  - Trên valid, R@10 và R@50 có đỉnh ở **uniform** (seed 42: .2017 / .4186). Chưa có biến thể nào vượt uniform trên valid một cách nhất quán.

### 4.3 Nhóm B: thang `r3_*` + `r2_c_noprop`, seed 3407

**Valid:**

| cell | ep | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | tail R@50 | rec_loss |
|---|---|---|---|---|---|---|---|---|
| r3_rel_uniform | 5/9 | .0322 | .1959 | .4253 | .0727 | .1013 | .0491 | 7.414 |
| r3_rel_fixed | 5/9 | .0331 | .1977 | .4235 | .0731 | .1020 | .0491 | 7.417 |
| r3_rel_attn | 4/8 | .0360 | .1997 | .4238 | .0758 | .1044 | .0443 | 7.271 |
| r3_age | 4/8 | .0360 | .1997 | .4241 | .0758 | .1044 | .0443 | 7.270 |
| r3_evidence | 4/8 | .0325 | .1983 | .4258 | .0742 | .1030 | .0479 | 7.276 |
| r3_evidence_noov | 5/9 | .0340 | .1968 | .4241 | .0744 | .1028 | .0527 | 7.392 |
| r3_evidence_noprop | 3/7 | .0366 | .1945 | .4258 | .0758 | .1034 | .0335 | 7.185 |
| r2_c_noprop | 3/7 | .0331 | .1881 | .4212 | .0712 | .0984 | .0311 | 7.201 |
| r2_rescope_ip | 4/8 | .0325 | .2012 | .4218 | .0747 | .1041 | .0419 | 7.284 |

**Test:**

| cell | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | tail R@10 | tail R@50 |
|---|---|---|---|---|---|---|---|
| r3_rel_uniform | .0272 | .1696 | .3468 | .0626 | .0875 | .0086 | .0373 |
| r3_rel_fixed | .0274 | .1686 | .3489 | .0626 | .0872 | .0067 | .0364 |
| r3_rel_attn | .0299 | .1703 | .3535 | .0646 | .0891 | .0077 | .0373 |
| r3_age | .0299 | .1705 | .3535 | .0646 | .0892 | .0077 | .0373 |
| r3_evidence | .0297 | .1715 | .3539 | .0642 | .0891 | .0115 | .0325 |
| r3_evidence_noov | .0299 | .1715 | .3520 | .0654 | .0902 | .0115 | .0364 |
| r3_evidence_noprop | .0307 | .1700 | .3537 | .0652 | .0896 | .0048 | .0258 |
| r2_c_noprop | .0290 | .1711 | .3570 | .0650 | .0897 | .0048 | .0220 |
| r2_rescope_ip | .0324 | .1700 | .3537 | .0671 | .0910 | .0077 | .0364 |

**Δ so với `r3_rel_uniform`** (chạy cùng đợt, cùng cấu hình), dạng test / valid:

| cell | ΔR@1 | ΔR@10 | ΔR@50 | ΔMRR@10 |
|---|---|---|---|---|
| r3_rel_fixed | +.0002 / +.0009 | −.0010 / +.0018 | +.0021 / −.0018 | .0000 / +.0004 |
| r3_rel_attn | **+.0027 / +.0038** | +.0007 / +.0038 | +.0067 / −.0015 | **+.0020 / +.0031** |
| r3_age | +.0027 / +.0038 | +.0009 / +.0038 | +.0067 / −.0012 | +.0020 / +.0031 |
| r3_evidence | +.0025 / +.0003 | +.0019 / +.0024 | +.0071 / +.0005 | +.0016 / +.0015 |
| r3_evidence_noov | +.0027 / +.0018 | +.0019 / +.0009 | +.0052 / −.0012 | +.0028 / +.0017 |
| r3_evidence_noprop | **+.0035 / +.0044** | +.0004 / −.0014 | +.0069 / +.0005 | **+.0026 / +.0031** |

Nhóm B chưa có `r2_uniform` seed 3407. Run đó chạy ở dropout 0.3, thuộc nhóm A.

#### Đường cong valid và đỉnh của từng biến thể (thang r3, seed 3407)

| cell | ep chọn / số ep | đỉnh R@1 | đỉnh R@10 | đỉnh R@50 | đỉnh MRR@10 | valid loss thấp nhất |
|---|---|---|---|---|---|---|
| r3_rel_uniform | 5 / 9 | .0369 @4 | .1959 @5 | .4253 @5 | .0756 @4 | 7.149 @2 |
| r3_rel_fixed | 5 / 9 | .0377 @4 | .1977 @5 | .4235 @5 | .0759 @4 | 7.147 @2 |
| r3_rel_attn | 4 / 8 | .0374 @5 | .1997 @4 | .4238 @4 | .0758 @4 | 7.163 @2 |
| r3_age | 4 / 8 | .0374 @5 | .1997 @4 | .4241 @4 | .0758 @4 | 7.163 @2 |
| r3_evidence | 4 / 8 | .0354 @3 | .1983 @4 | **.4258 @4** | .0758 @3 | 7.173 @2 |
| r3_evidence_noov | 5 / 9 | **.0383 @7** | **.2000 @4** | .4241 @5 | **.0776 @3** | 7.167 @2 |
| r3_evidence_noprop | 3 / 7 | .0366 @3 | .1959 @5 | **.4258 @3** | .0758 @3 | 7.175 @2 |
| r2_c_noprop | 3 / 7 | .0366 @4 | .1942 @4 | .4212 @3 | .0754 @4 | 7.148 @1 |
| r2_rescope_ip | 4 / 8 | .0377 @6 | .2012 @4 | .4218 @4 | .0750 @6 | 7.122 @2 |

Dưới đây là cell tốt nhất theo từng metric. Cột "r3 tốt nhất" chỉ xét thang r3; cột "cả bảng" tính cả `r2_c_noprop` và `r2_rescope_ip`.

| metric | test, r3 tốt nhất | test, cả bảng | valid (ep chọn), cả bảng |
|---|---|---|---|
| R@1 | evidence_noprop .0307 | **ip .0324** | evidence_noprop .0366 |
| R@10 | evidence = noov .1715 | evidence = noov .1715 | ip .2012 |
| R@50 | evidence .3539 | **c_noprop .3570** | evidence = noprop .4258 |
| MRR@10 | noov .0654 | **ip .0671** | rel_attn = age = noprop .0758 |
| NDCG@10 | noov .0902 | **ip .0910** | rel_attn = age .1044 |
| tail R@10 | evidence = noov .0115 | evidence = noov .0115 | — |
| tail R@50 | rel_uniform = rel_attn = age .0373 | như cột trước | noov .0527 |

**Nhận xét bảng 4.3, đi theo từng bậc của thang:**

- **Kiểu đường cong chung.** Giống nhóm B ở 4.2: valid loss thấp nhất ở epoch 2 (riêng `c_noprop` ở epoch 1), và R@50 đạt đỉnh ở epoch 3–5. Sau epoch 2, mô hình overfit về loss trong khi thứ hạng còn cải thiện thêm 1–3 epoch.

- **`r3_rel_uniform` → `r3_rel_fixed`: thêm decay 0.7^tuổi vào β. Gần như không đổi gì.**
  - Valid loss trùng đến 3 chữ số (7.149 so với 7.147), đỉnh rơi cùng epoch (R@1 @4, R@50 @5), và test chênh ≤ .0021.
  - Điều này đáng chú ý: β bị đổi rất mạnh (cửa sổ tuổi 3 chỉ còn nặng .34 so với cửa sổ mới nhất), vậy mà mô hình gần như không phản ứng.
  - Cách hiểu hợp lý: việc cân lại các cửa sổ **cùng nội dung sau conv** ít ảnh hưởng tới hàng C. Sau 2 lớp conv, các r_e của những cửa sổ chồng lấn (w = 3) đã rất giống nhau.
  - Hai cell này là đáy của thang trên test: R@1 khoảng .027, R@50 .347–.349. R@50 thấp hơn mọi cell còn lại khoảng .005.

- **`r3_rel_uniform` → `r3_rel_attn`: thêm relevance. Đây là bước nhảy lớn nhất của thang.**
  - Trên test: R@1 +.0027, MRR +.0020, R@50 +.0067. Trên valid, R@1 và MRR cùng chiều (+.0038 / +.0031).
  - Đường cong valid khác hẳn ở đầu bảng: ngay từ epoch 0, R@1 đã là .0264 so với .0244, và tới epoch 2 là .0357 so với .0351. Relevance giúp từ sớm.
  - Đỉnh R@10 trên valid đến sớm hơn một epoch (epoch 4 so với 5), và early stop cũng sớm hơn (8 epoch so với 9).
  - Khác với fixed, relevance đổi trọng số **theo nội dung của lượt hiện tại**, nên nó khác nhau giữa các mẫu. Đây là loại tín hiệu mà decay chung không tạo ra được.

- **`r3_rel_attn` → `r3_age`: thêm một tốc độ quên học được. Không đổi gì.**
  - Mọi metric valid ở mọi epoch trùng đến 3–4 chữ số, và test chênh ≤ .0002.
  - Tham số rate_logit gần như không nhận gradient nên đứng yên ở khởi tạo, quên khoảng 1.8% mỗi lượt.
  - Bậc này không cho thông tin gì về retention, ngoài việc loss không có động lực để học một decay chung.

- **`r3_age` → `r3_evidence`: forget gate theo nội dung. Học ra "quên gần hết" (ρ ở tuổi 1 ≈ .06), nhưng kết quả gần như ngang `rel_attn`.**
  - Trên valid, đỉnh R@1 và MRR của evidence đến **sớm hơn** (epoch 3), và R@1 thấp hơn ở epoch được chọn (.0325 so với .0360).
  - Ngược lại, valid R@50 của evidence là cao nhất thang (.4258, ngang noprop).
  - Trên test, evidence ngang `rel_attn` ở R@1 và MRR (−.0002 / −.0004) và nhỉnh hơn ở R@10 (+.0012).
  - Tail R@10 cao nhất thang (.0115), nhưng tail R@50 lại giảm (.0325 so với .0373). Số mẫu tail rất nhỏ, nên hai chênh lệch ngược chiều này nhiều khả năng là nhiễu.
  - Kết luận: chuyển từ "không quên" sang "chỉ giữ cửa sổ mới nhất" **không đổi chất lượng xếp hạng**. Retention không phải nút điều khiển có ý nghĩa ở ReDial.

- **`r3_evidence` → `r3_evidence_noov`: bỏ đặc trưng nhắc lại. Không tệ hơn, có phần tốt hơn.**
  - Cell này train lâu nhất (9 epoch). Trên valid nó đạt **đỉnh R@1 (.0383 @7), đỉnh R@10 (.2000 @4) và đỉnh MRR (.0776 @3) cao nhất của cả bảng**.
  - Nhưng các đỉnh này rơi vào những epoch khác nhau, và checkpoint chọn theo R@50 là epoch 5, nên bỏ lỡ cả ba.
  - Trên test, noov có MRR và NDCG cao nhất thang r3 (.0654 / .0902), và R@10 ngang evidence.
  - Đặc trưng ov không được mô hình dùng. Bỏ nó đi còn làm đường cong đầu bảng ổn định hơn: R@1 trên valid nằm trong khoảng .034–.038 từ epoch 2 đến epoch 8, không có cú rơi sau đỉnh.

- **`r3_evidence` → `r3_evidence_noprop`: tắt conv ở C. Đầu bảng tốt nhất của thang r3, đổi lại tail tệ nhất.**
  - Hội tụ nhanh nhất: đỉnh R@1, R@50 và MRR **cùng rơi vào epoch 3**, đúng epoch được chọn. Đây là cell duy nhất mà checkpoint trùng đỉnh đầu bảng.
  - Trên test, R@1 .0307 là cao nhất thang r3, nhưng tail R@10/R@50 rơi xuống .0048 / .0258, thấp nhất thang.
  - Không có conv, mỗi node chỉ còn embedding của chính nó (W x_v + b). Phim ít phổ biến mất đường nhận tín hiệu từ các node cùng cửa sổ, nên tail tụt.

- **Đối chứng `r2_c_noprop`** (tắt conv, readout mean theo node, không relevance):
  - Valid loss đạt đáy sớm nhất (epoch 1) và có R@10 trên valid thấp nhất bảng (.1881 ở epoch chọn, đỉnh .1942).
  - Trên test, nó có **R@50 cao nhất cả bảng (.3570)** và tail R@50 thấp nhất (.0220).
  - So với `evidence_noprop` (cũng tắt conv nhưng có relevance và readout theo relation): c_noprop thấp hơn ở R@1 (.0290 so với .0307) nhưng cao hơn ở R@50.
  - Khi đã tắt conv, relevance vẫn góp thêm phần đầu bảng, tương tự bước rel_uniform → rel_attn.

- **`r2_rescope_ip`, dòng tham chiếu:**
  - Trên valid, ip có R@10 cao nhất (.2012) nhưng R@1 ở epoch chọn chỉ trung bình (.0325). Đỉnh R@1 của nó đến muộn (.0377 @6).
  - Trên test, ip đứng **đầu bảng ở R@1, MRR và NDCG** (.0324 / .0671 / .0910), vượt cell r3 tốt nhất khoảng .0017 / .0017 / .0008.
  - Khác biệt so với `evidence_noprop` là ip vẫn giữ một chút propagation (α ≈ .01, không phải 0) và readout theo node với trọng số recency. Mới có một seed nên chưa nói được khác biệt này có thật không.

**Đỉnh nằm ở đâu:**

| metric | đỉnh trên test thuộc về |
|---|---|
| R@1, MRR, NDCG | `r2_rescope_ip`; trong thang r3 là `evidence_noprop` (R@1) và `noov` (MRR/NDCG) |
| R@10 | `evidence` / `noov` |
| R@50 | `r2_c_noprop` |
| tail | các cell **có** conv ở C (`rel_*`, `age`, `evidence`, `noov`) |

Trên valid, đỉnh đầu bảng thuộc về `noov` nhưng không được checkpoint chọn. **Không có cell nào đứng đầu mọi metric.** Các bậc tạo ra khác biệt là **relevance** và **tắt conv ở C**; mọi bậc retention (fixed, age, evidence, ov) đều cho chênh lệch ≈ 0.

### 4.4 Độ nhiễu để đọc các Δ

- **Nhiễu lấy mẫu của một run.** Test có khoảng 4.8k mẫu, nên SE(R@1) ≈ √(.03·.97/4787) ≈ .0025. Valid có khoảng 3.5k mẫu, nên SE(R@1) ≈ .003. So ghép cặp trên cùng mẫu thì nhiễu nhỏ hơn con số này, nhưng không nhỏ hơn nhiều.
- **Nhiễu giữa seed** (uniform, seed 42 với 123, cùng cấu hình): chênh R@1 .0040, R@10 .0046, MRR .0031.
- **Quy ước đọc:** một Δ của 1 seed nhỏ hơn khoảng .004 ở R@1/R@10 chưa phải là bằng chứng.

---

## 5. Nhận xét

### 5.1 Thiết kế → hành vi gate: loss đầu-cuối không dạy được retention có chọn lọc

Ba họ cơ chế đã thử, mỗi họ có cách tham số hoá khác hẳn nhau:

- σ chỉ làm giảm message (v1);
- σ nội suy với embedding của chính node (v2), có hoặc không có prior;
- forget gate tích luỹ, có mốc (evidence).

Cả ba đều hội tụ về một điểm cực: v1 mở hết, v2 đóng hết, evidence quên hết. Không lần nào gate học được việc giữ một window cũ và bỏ một window khác cùng tuổi. Std trong mẫu luôn ≈ std giữa mẫu, và đặc trưng nhắc lại (`noov`) không được dùng.

Hướng hội tụ được quyết định bởi chỗ gate tác động, không phải bởi nó học được gì về hội thoại:

- Gate chạm vào conv thì loss dùng nó để **tắt propagation của C**. Dấu hiệu: valid loss tăng từ epoch 3, và C là một nguồn overfit.
- Gate chỉ chạm vào readout (evidence) thì loss dùng nó để **dồn readout về window mới nhất**.

Ở cả hai trường hợp, gate là một nút điều chỉnh độ mạnh/độ gần của scope C, không phải cơ chế ghi nhớ. Với tín hiệu yếu như vậy (C là 3/7 hàng của MHA, 88% target cold, chỉ 18% mẫu có nhãn lẫn), không có lý do để kỳ vọng loss recommendation đủ sức tách các window.

### 5.2 Kết quả → chỉ có một hiệu ứng lặp lại: "C yếu đi / gần hơn" làm tăng đầu bảng

| bằng chứng | ΔR@1 test | ΔMRR test |
|---|---|---|
| ip so với uniform, nhóm A, seed 3407 | +.0042 | +.0029 |
| ip so với uniform, nhóm B, seed 42 | +.0025 | +.0017 |
| ip so với uniform, nhóm B, seed 123 | +.0052 | +.0038 |
| i so với uniform, nhóm A | +.0044 | +.0029 |
| evidence_noprop so với rel_uniform | +.0035 | +.0026 |
| rel_attn so với rel_uniform | +.0027 | +.0020 |

- Hướng nhất quán: 6/6 so sánh dương. Độ lớn khoảng +.003–.005 R@1 và +.002–.004 MRR, xấp xỉ 1–2 SE mỗi lần. Gộp lại là bằng chứng vừa phải rằng *giảm/tắt propagation ở C và dồn readout về gần* có lợi cho đầu bảng.
- Đổi lại, R@10 hơi giảm (ip: −.0015 đến −.0050 ở 3/3 cặp) và **tail giảm**. Tail R@50 của hai cell noprop là .022 / .026, so với khoảng .036–.041 ở các cell khác. Propagation qua window có vẻ là đường đưa tín hiệu tới item hiếm.
- **Decay cố định** (fixed γ = .7, cả node lẫn relation) không tạo hiệu ứng đầu bảng. Nó chỉ cho một tăng nhỏ ở R@50 test (+.0012 / +.0040 / +.0025), và tăng này không thấy trên valid.
- **Retention theo đúng nghĩa** (chọn window cũ nào để giữ) không đóng góp: `r3_age` ≈ `r3_evidence` ≈ `r3_rel_attn`, dù ba cell có ρ rất khác nhau.

### 5.3 Valid và test không đồng thuận

- Hiệu ứng đầu bảng của ip có trên **test** ở cả 3 cặp, nhưng **không có trên valid**. ΔR@1 valid của ip là −.0003 và −.0041 ở seed 42/123; ở nhóm A thì không có valid R@1 của uniform để so. Ngược lại, các cell r3 có relevance hoặc noprop thì valid và test cùng chiều: rel_attn +.0038 / +.0027, noprop +.0044 / +.0035.
- Valid R@50, chỉ số dùng để chọn checkpoint, gần như phẳng giữa mọi cell: .415–.426, chênh khoảng 1 SE. Epoch được chọn (3–5) vì vậy gần như ngẫu nhiên giữa các cell.
- Gap valid→test lớn và giống nhau ở mọi cell: khoảng .03 R@10 và .07 R@50. Gap này do cấu hình chung (rec-only, window_dedupe), không do retention.
- **Hệ quả:** không thể dùng valid để tune retention trong cấu hình này. Một hiệu ứng chỉ có trên test thì cần nhiều seed hơn mới báo cáo được. Riêng ip đã có 3 cặp test cùng dương.

### 5.4 Ý nghĩa với claim RQ2

- Claim "*learned retention tốt hơn uniform / latest / fixed decay*" **chưa được hỗ trợ** trên ReDial. Biến thể tốt nhất (ip) tốt hơn ở R@1/MRR nhưng kém hơn ở R@10 và tail. Cơ chế của nó cũng không phải retention: gate đã sập, và hiệu ứng tương đương với "tắt propagation ở C + readout recency".
- Có thể nói một cách trung thực: *trên ReDial, độ hữu ích của ngữ cảnh tập trung ở turn gần nhất; retention học được rút về recency; propagation qua các window cũ đánh đổi độ chính xác đầu bảng để lấy độ phủ tail*. Đây là một phát hiện phân tích hợp lệ cho RQ2, nhưng không phải phát hiện "phương pháp mới thắng".
- Nếu muốn giữ claim retention, phải tìm nơi có không gian cho nó. TG-ReDial có 41% mẫu nhãn lẫn, và window tuổi ≥7 được nhắc lại có utility .475 so với .154 nếu không. Nhưng công thức TG hiện tại hỏng: R@10 khoảng .028, so với .031 của MHIM. Phải sửa công thức trước.

### 5.5 Những điểm thiết kế đáng giữ lại

- **Relevance tới turn hiện tại** (W_q u_T · W_k h_e) là thành phần học được duy nhất cho tín hiệu cùng chiều trên cả valid và test (+.003 R@1). Nó cũng khớp với probe, nơi rel_now là đặc trưng mạnh nhất (AUC .747).
- **Mốc ρ = 1 + softmax** loại được hiện tượng "α là thang theo mẫu" của RetentionGate. Thiết kế đúng hơn về mặt hình thức, dù không cải thiện metric.
- **Tách conv khỏi retention**: retention chỉ ở readout, còn việc C có propagate hay không là một lựa chọn kiến trúc riêng (`c_propagate`). Cách tách này làm rõ được nguồn gốc lợi ích của ip, điều mà RetentionGate che lấp.

---

## 6. Hướng tiếp theo

1. **Chốt bằng chứng cho hiệu ứng đầu bảng** ở nhóm B: chạy lại uniform seed 3407 ở dropout 0.2, và chạy seed 42/123 cho `r2_c_noprop`, `r3_evidence_noprop`, `r3_rel_attn`, `r3_rel_uniform`. Lệnh ở `docs/r3_results_hredial.md` mục 7.
2. **Báo cáo đánh đổi đầu bảng ↔ tail**, không chỉ R@10. Nếu noprop bằng ip thì dùng noprop vì đơn giản hơn và nhanh hơn: khoảng 23 so với 27 phút/epoch.
3. **Nếu vẫn theo đuổi retention**, cần tín hiệu học trực tiếp hơn loss rec, ví dụ một auxiliary loss với nhãn utility của window (như trong probe). Ngoài ra cần thử trên TG sau khi sửa công thức.
4. **Kiểm soát chọn checkpoint**: valid R@50 không phân biệt được các cell. Cân nhắc dừng theo valid MRR/NDCG, hoặc cố định số epoch cho mọi cell.

---

## Phụ lục: thuật ngữ

| thuật ngữ | nghĩa |
|---|---|
| turn / lượt | một câu nói của seeker hoặc system |
| field | loại node: item (phim), entity (khái niệm trong DBpedia), word (từ khoá trong ConceptNet) |
| scope C / P / G | ngữ cảnh hiện tại / lịch sử cá nhân / thống kê cộng đồng (mục 0.1) |
| hyperedge, window, relation | trong tài liệu này là cùng một thứ: nhóm node của w = 3 lượt liên tiếp |
| tuổi (age) | số lượt từ khi cửa sổ kết thúc đến lượt hiện tại; cửa sổ mới nhất có tuổi 0 |
| conv / propagation | node trong cùng hyperedge trao đổi thông tin |
| readout / pooling | gộp node thành vector của một hàng |
| α (alpha) | trọng số retention của RetentionGate cho mỗi cửa sổ, trong (0, 1) |
| ρ (rho) | độ còn nhớ của EvidenceRetention: tích của (1 − mức quên) qua các lượt đến sau; cửa sổ mới nhất có ρ = 1 |
| β (beta) | trọng số softmax cuối cùng của mỗi cửa sổ ở readout (evidence) |
| relevance | độ liên quan giữa một cửa sổ và lượt hiện tại, học bằng attention |
| `scale` / `interp` | hai cách gate conv: nhân message với α / nội suy giữa message và embedding của chính node |
| prior decay | khởi tạo α = 0.75^tuổi theo thống kê dữ liệu |
| cold target | phim đúng chưa từng xuất hiện trong ngữ cảnh (88% ở ReDial) |
| nhãn lẫn | mẫu có cả cửa sổ hữu ích lẫn không hữu ích, tức việc chọn cửa sổ có thể tạo khác biệt |
| ablation | bỏ hoặc thay một thành phần để đo đóng góp của nó |
| Δ ghép cặp | hiệu metric giữa hai cell cùng seed và cùng cấu hình |
| SE | sai số chuẩn do lấy mẫu; với R@1 trên test khoảng .0025 |
| đầu bảng / độ phủ / tail | R@1, MRR, NDCG / R@10, R@50 / các metric trên đuôi dài |
