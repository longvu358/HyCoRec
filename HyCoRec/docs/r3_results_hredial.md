# Kết quả thang ablation retention `r3_*` + `r2_c_noprop` (HReDial, seed 3407)

Log: `log/20261022-234858/`. Checkpoint: `save/cpg_hredial_<cell>_seed3407_20261022-234858.pth`.
Dump ρ: `results/cpg/dump/hredial_r3_evidence_seed3407.jsonl` (qua `run_cpg.py analyze --force`).

---

## 1. Thiết lập chung

Mọi cell dưới đây dùng chung một cấu hình. Chỉ các khoá liệt ở mục 2 là khác nhau.

| nhóm | giá trị |
|---|---|
| dữ liệu | HReDial, `scale: 1` (toàn bộ dữ liệu) |
| scope | C + P + G, `fusion_mode: rows7` (7 hàng khoá [C_I, C_E, C_W, P_I, P_E, P_W, G], MHA với truy vấn là context entity) |
| scope C | sliding window `context_window_w: 3`, `window_dedupe: true` (gộp các window trùng nội dung) |
| scope P / G | `k_hist: 40`; G là Â^G từ `data/collective/hredial_full`, `g_query` mặc định (`ctx+hist`) |
| hypergraph conv | `hconv_layers: 2`, `hconv_dropout: 0.2`, `kg_emb_dim: 128` |
| huấn luyện | chỉ phần recommendation (`rec_only: true`), tối đa 10 epoch, batch 256, AdamW lr 5e-4, wd 0.1, clip 1, ReduceLROnPlateau (factor 0.5, patience 0) |
| chọn checkpoint | early stop theo **valid recall@50**, impatience 3. Test chạy trên epoch có valid R@50 cao nhất |
| seed | 3407 (một seed) |
| phần cứng | 4 V100, 4 job chạy song song. Thời gian/epoch vì vậy chỉ mang tính tương đối |

---

## 2. Giải thích các config

### 2.1 Cơ chế mới: `retention: evidence` (`cpc.EvidenceRetention`)

Với mỗi field f ∈ {I, E, W}, scope C có một tập window hyperedge e. Mỗi window có tuổi `age(e)`, tức số turn tính từ window e đến turn hiện tại T.

1. **Biểu diễn relation**: r_e^f = mean các node của e sau hypergraph conv. Conv ở C không bị gate. Nếu `c_propagate: false`, conv ở C bị tắt và mỗi node chỉ còn lin(x)+b.
2. **Relevance**: s_e = (W_q u_T)·(W_k h_e)/√d. Ở đây u_T là biểu diễn của turn hiện tại, còn h_e là biểu diễn của window. W_k được khởi tạo bằng 0, nên ban đầu s_e = 0.
3. **Forget gate**: với mỗi turn t đến **sau** window e, f_{e,t} = σ(MLP([h_e; u_t; h_e⊙u_t; ov(e,t)]) − 4). Trong đó ov(e,t) là tỉ lệ node của e được nhắc lại ở turn t. Khi khởi tạo, mỗi turn quên khoảng 1.8%.
4. **Retention**: log ρ_e = Σ_{t>t_e} log(1 − f_{e,t}). Window mới nhất có ρ = 1 và làm mốc.
5. **Readout**: β_e = softmax_e(s_e + log ρ_e), rồi c^C_f = Σ_e β_e r_e^f.

### 2.2 Các cell

Mỗi cell r3 bỏ đúng một thành phần so với `r3_evidence`. Tất cả đều có `window_dedupe: true` và `retention: evidence`, cộng thêm các khoá ghi ở cột thứ hai.

| cell | khoá khác | relevance | ρ (retention) | C propagate | ý nghĩa |
|---|---|---|---|---|---|
| `r3_rel_uniform` | `retention_forget: none`, `retention_relevance: false` | ✗ | 1 | ✓ | readout theo relation, mean thường. Là gốc của thang |
| `r3_rel_fixed` | `retention_forget: const`, `retention_gamma: 0.7`, `retention_relevance: false` | ✗ | 0.7^age | ✓ | decay cố định, không học |
| `r3_rel_attn` | `retention_forget: none` | ✓ | 1 | ✓ | chỉ có relevance tới turn hiện tại, không quên |
| `r3_age` | `retention_forget: age` | ✓ | (1−f)^age, học **một** f chung | ✓ | relevance + tốc độ quên học được, chỉ phụ thuộc tuổi |
| **`r3_evidence`** | (mặc định) | ✓ | forget gate theo nội dung các turn sau + overlap | ✓ | **cơ chế đầy đủ** |
| `r3_evidence_noov` | `retention_overlap: false` | ✓ | forget gate không có ov(e,t) | ✓ | kiểm tra đặc trưng "nhắc lại" |
| `r3_evidence_noprop` | `c_propagate: false` | ✓ | như evidence | ✗ | evidence + tắt conv ở C |

Có ba cell đối chứng không thuộc thang r3. Chúng dùng readout theo **node**, không theo relation:

| cell | khoá | ý nghĩa |
|---|---|---|
| `r2_uniform` | `window_dedupe: true` (không retention) | baseline: mean các node của scope C |
| `r2_c_noprop` | `retention: const`, `retention_const: 0.0`, `retention_gate_mode: interp` | α ≡ 0 với cổng nội suy, nên C **không propagate** (node giữ lin(x)+b). Readout không có trọng số |
| `r2_rescope_ip` | `retention: learned`, `retention_gate_mode: interp`, `retention_prior: decay`, `retention_gamma: 0.75` | cơ chế cũ (RetentionGate): α học được gate conv và trọng số readout max-α |

---

## 3. Kết quả

### 3.1 Valid (epoch được chọn theo valid R@50)

`ep` là epoch được chọn trên tổng số epoch đã chạy (đánh số từ 0).

| cell | ep | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | tail R@10 | tail R@50 | rec_loss |
|---|---|---|---|---|---|---|---|---|---|
| r3_rel_uniform | 5/9 | .0322 | .1959 | .4253 | .0727 | .1013 | .0132 | .0491 | 7.414 |
| r3_rel_fixed | 5/9 | .0331 | .1977 | .4235 | .0731 | .1020 | .0144 | .0491 | 7.417 |
| r3_rel_attn | 4/8 | .0360 | .1997 | .4238 | .0758 | .1044 | .0144 | .0443 | 7.271 |
| r3_age | 4/8 | .0360 | .1997 | .4241 | .0758 | .1044 | .0144 | .0443 | 7.270 |
| r3_evidence | 4/8 | .0325 | .1983 | .4258 | .0742 | .1030 | .0120 | .0479 | 7.276 |
| r3_evidence_noov | 5/9 | .0340 | .1968 | .4241 | .0744 | .1028 | .0156 | .0527 | 7.392 |
| r3_evidence_noprop | 3/7 | .0366 | .1945 | .4258 | .0758 | .1034 | .0096 | .0335 | 7.185 |
| r2_c_noprop | 3/7 | .0331 | .1881 | .4212 | .0712 | .0984 | .0084 | .0311 | 7.201 |
| *r2_rescope_ip* | 4/8 | .0325 | .2012 | .4218 | .0747 | .1041 | .0168 | .0419 | 7.284 |
| *r2_uniform* | 5 | – | .1977 | .4226 | – | – | – | – | – |

Dòng `r2_rescope_ip` lấy từ `log/20261022-021023`. Dòng `r2_uniform` lấy từ `docs/retention_log.md` vòng 3, vì log `20261021-222655` không có ở local.
**Run `r2_uniform` seed 3407 dùng `hconv_dropout: 0.3`**, khác với mọi cell khác (0.2), nên không so trực tiếp được (xem mục 6).

### 3.2 Test (checkpoint ở epoch trên)

| cell | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | tail R@10 | tail R@50 | phút/epoch |
|---|---|---|---|---|---|---|---|---|
| r3_rel_uniform | .0272 | .1696 | .3468 | .0626 | .0875 | .0086 | .0373 | 25.7 |
| r3_rel_fixed | .0274 | .1686 | .3489 | .0626 | .0872 | .0067 | .0364 | 25.8 |
| r3_rel_attn | .0299 | .1703 | .3535 | .0646 | .0891 | .0077 | .0373 | 27.3 |
| r3_age | .0299 | .1705 | .3535 | .0646 | .0892 | .0077 | .0373 | 28.2 |
| r3_evidence | .0297 | .1715 | .3539 | .0642 | .0891 | .0115 | .0325 | 27.7 |
| r3_evidence_noov | .0299 | .1715 | .3520 | .0654 | .0902 | .0115 | .0364 | 27.5 |
| r3_evidence_noprop | **.0307** | .1700 | .3537 | .0652 | .0896 | .0048 | .0258 | 29.7 |
| r2_c_noprop | .0290 | .1711 | **.3570** | .0650 | .0897 | .0048 | .0220 | 23.0 |
| *r2_rescope_ip* | *.0324* | *.1700* | *.3537* | *.0671* | *.0910* | *.0077* | *.0364* | ~27 |
| *r2_uniform (dropout 0.3)* | *.0265* | *.1692* | *.3537* | *.0624* | *.0872* | *.0057* | *.0412* | ~21 |

### 3.3 Δ test (seed 3407)

Có hai mốc so sánh. Mốc B là `r3_rel_uniform`, chạy cùng đợt nên điều kiện giống hệt. **Mốc B là mốc chính.**
Mốc A là `r2_uniform` seed 3407, nhưng run này dùng `hconv_dropout` 0.3 thay vì 0.2. Các Δ theo mốc A vì vậy lẫn cả hiệu ứng của dropout, chỉ nên xem để tham khảo.

| cell | ΔR@1 (A / B) | ΔR@10 (A / B) | ΔR@50 (A / B) | ΔMRR@10 (A / B) | ΔNDCG@10 (A / B) |
|---|---|---|---|---|---|
| r3_rel_uniform | +.0007 / — | +.0004 / — | −.0069 / — | +.0002 / — | +.0003 / — |
| r3_rel_fixed | +.0009 / +.0002 | −.0006 / −.0010 | −.0048 / +.0021 | +.0002 / .0000 | .0000 / −.0003 |
| r3_rel_attn | +.0034 / +.0027 | +.0011 / +.0007 | −.0002 / +.0067 | +.0022 / +.0020 | +.0019 / +.0016 |
| r3_age | +.0034 / +.0027 | +.0013 / +.0009 | −.0002 / +.0067 | +.0022 / +.0020 | +.0020 / +.0017 |
| r3_evidence | +.0032 / +.0025 | +.0023 / +.0019 | +.0002 / +.0071 | +.0018 / +.0016 | +.0019 / +.0016 |
| r3_evidence_noov | +.0034 / +.0027 | +.0023 / +.0019 | −.0017 / +.0052 | +.0030 / +.0028 | +.0030 / +.0027 |
| r3_evidence_noprop | +.0042 / +.0035 | +.0008 / +.0004 | .0000 / +.0069 | +.0028 / +.0026 | +.0024 / +.0021 |
| r2_c_noprop | +.0025 / — | +.0019 / — | +.0033 / — | +.0026 / — | +.0025 / — |
| r2_rescope_ip | +.0058 / — | +.0008 / — | .0000 / — | +.0047 / — | +.0038 / — |

Để đặt các Δ vào đúng tỉ lệ, đây là độ nhiễu của `r2_uniform`. Cột "3 seed" gộp seed 3407 (dropout 0.3) với seed 42/123 (dropout 0.2), nên lẫn hai cấu hình. Cột "2 seed" chỉ gồm 42/123 và đúng cấu hình 0.2:

| metric | 3 seed: trung bình ± std | 2 seed (42 / 123, dropout 0.2): trung bình, chênh giữa hai seed |
|---|---|---|
| R@1 | .0281 ± .0024 | .0289, chênh .0040 |
| R@10 | .1688 ± .0023 | .1686, chênh .0046 |
| R@50 | .3515 ± .0020 | .3505, chênh .0011 |
| MRR@10 | .0629 ± .0016 | .0632, chênh .0031 |
| NDCG@10 | .0875 ± .0007 | .0877, chênh .0014 |

**`r2_rescope_ip` so với `r2_uniform`, ghép cặp, chỉ dùng các seed cùng dropout 0.2:**

| seed | ΔR@1 | ΔR@10 | ΔR@50 | ΔMRR@10 | ΔNDCG@10 |
|---|---|---|---|---|---|
| 42 | +.0025 | −.0015 | −.0010 | +.0017 | +.0010 |
| 123 | +.0052 | −.0050 | +.0094 | +.0038 | +.0019 |

Hiệu ứng đầu bảng của ip vẫn còn khi bỏ seed 3407 bị lệch cấu hình: R@1 tăng trung bình +.0039, MRR +.0028, cả hai đều dương ở cả 2 seed. R@10 giảm ở cả 2 seed.
ip seed 3407 không có uniform cùng cấu hình để ghép cặp. R@1 của ip qua 3 seed là .0327 ± .0007.

### 3.4 Retention ρ học được (`r3_evidence`, dump trên test)

| tuổi | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ≥8 |
|---|---|---|---|---|---|---|---|---|---|
| n window | 4772 | 4497 | 3816 | 3334 | 2442 | 1999 | 1343 | 1045 | 2107 |
| ρ trung bình | 1.000 | .057 | .012 | .003 | .003 | .0002 | .0008 | .0002 | 0 |

- 77.3% window có ρ < .05. Trung vị là 0, phân vị 75% là .019. 19% window có ρ > .9, gần như toàn bộ là các window tuổi 0 (mốc).
- corr(tuổi, ρ) = −.51.
- Ngay ở turn đầu tiên đến sau, forget gate đã quên khoảng 94%. Như vậy readout C gần như **chỉ dùng window mới nhất**.

---

## 4. Phân tích

### 4.1 Retention (chọn giữ window cũ nào) không tạo khác biệt đo được

- `r3_age` gần như trùng `r3_rel_attn`: valid ở từng epoch khớp đến 3–4 chữ số, test chênh ≤ .0002. Tốc độ quên học được gần như không rời khỏi giá trị khởi tạo (~1.8% mỗi turn), tức là nó không nhận được gradient có ý nghĩa.
- `r3_evidence` đi theo hướng ngược lại: nó quên khoảng 94% mỗi turn. Dù vậy, test vẫn ngang `r3_rel_attn` (ΔR@1 −.0002, ΔR@10 +.0012, ΔMRR −.0004), đều nằm trong nhiễu.
- Hai chế độ retention trái ngược nhau ("giữ gần hết" và "chỉ giữ window mới nhất") cho cùng một kết quả. Vậy ở ReDial, việc chọn giữ window cũ nào không ảnh hưởng metric.
- `r3_evidence_noov` ≈ `r3_evidence`, nên đặc trưng nhắc lại không được dùng. Điều này khớp với việc forget gate bị bão hoà: khi đã quên gần hết thì overlap không còn tác dụng.
- `r3_rel_fixed` ≈ `r3_rel_uniform`: decay cố định γ = .7 cũng không có tác dụng, giống kết luận của vòng 3 với `r2_fixed_g07`.
- Kết quả này khớp với phân tích trước khi train (`evidence_probe.py`): chỉ 18% mẫu ReDial có nhãn lẫn giữa các window, nên trần lợi ích của việc chọn window vốn đã nhỏ.

### 4.2 Relevance tới turn hiện tại là thành phần duy nhất trong r3 có dấu hiệu

- So `r3_rel_attn` với `r3_rel_uniform` trong cùng đợt: ΔR@1 +.0027, ΔMRR +.0020, ΔNDCG +.0016, ΔR@50 +.0067.
- Valid cùng chiều: R@1 .0360 so với .0322, MRR .0758 so với .0727.
- Độ lớn khoảng 1 std của R@1 và MRR, và mới có 1 seed. Đây là **dấu hiệu, chưa phải kết luận**.

### 4.3 Tắt propagation ở C có ích, độc lập với retention

- `r2_c_noprop` so với `r2_uniform`: R@1 +.0025, MRR +.0026, NDCG +.0025. Đây cũng là cell duy nhất tăng cả R@50 (+.0033).
- `r3_evidence_noprop` có R@1 test cao nhất trong thang r3 (.0307) và valid R@1 cao nhất (.0366).
- Ghép với vòng 5 (α của ip sập về khoảng .01, tức C của ip gần như không propagate): phần lớn lợi ích đầu bảng của `r2_rescope_ip` nhiều khả năng đến từ (i) C không propagate, cộng với (ii) readout dồn mạnh về turn gần nhất, **không phải** từ retention.
- **Cái giá là tail**. Tail R@50 test của hai cell noprop giảm mạnh: .0220 và .0258, so với khoảng .036–.041 ở các cell khác. Tail R@10 cũng giảm, .0048 so với .008–.011. Propagation qua window có vẻ là đường đưa tín hiệu tới các item ít phổ biến. Lưu ý số mẫu tail nhỏ nên tail metric rất nhiễu.

### 4.4 Cơ chế mới chưa vượt được `r2_rescope_ip`

- Ở R@1 và MRR, ip seed 3407 (.0324 / .0671) vẫn cao hơn mọi cell r3. Cell r3 tốt nhất là `r3_evidence_noprop` với .0307 / .0652.
- Ở R@10 và R@50, các cell r3 ngang hoặc hơn ip một chút. `r3_evidence` và `noov` đạt R@10 .1715, so với .1700 của ip.
- ip có R@1 3 seed rất ổn định (std .0007). Mức chênh .0017 so với `r3_evidence_noprop` có thể là thật, nhưng cần seed 42/123 của r3 mới kết luận được.

### 4.5 Readout theo relation, khi chỉ lấy mean, làm giảm R@50

- `r3_rel_uniform` và `r3_rel_fixed` có R@50 test thấp hơn `r2_uniform` (readout theo node) khoảng .005–.007. Mức này vượt std R@50 (.0020).
- Valid lại không cho thấy điều đó (.4253 so với .4226). Khi thêm relevance (`rel_attn`), R@50 test trở lại ngang uniform.

### 4.6 Valid không dự đoán tốt test

- Valid R@50 (chỉ số dùng để chọn checkpoint) của mọi cell nằm trong khoảng .421–.426, tức chênh ±.0025, ngang mức nhiễu. Việc chọn checkpoint gần như ngẫu nhiên giữa các epoch 3–5.
- Gap valid→test lớn và giống nhau ở mọi cell: khoảng .03 R@10 và .07 R@50. Gap này vì vậy không do retention mà do cấu hình chung, như đã ghi ở vòng 3.
- Thứ tự giữa các cell trên valid không khớp test. Ví dụ, `r3_rel_attn` có valid R@1 .0360 nhưng test chỉ .0299, còn ip có valid R@1 .0325 nhưng test .0324.

---

## 5. Kết luận cho RQ2 (ReDial)

1. Retention học được (cả cơ chế cũ lẫn cơ chế evidence) đều rút về recency. Trên ReDial, các biến thể retention không khác nhau ở bất kỳ metric nào.
2. Phần có ích ở scope C là: **C không propagate qua window** (đầu bảng tăng, tail giảm), cộng với **trọng số theo relevance hoặc recency ở readout**. Bản thân cơ chế giữ/quên không đóng góp.
3. Nếu muốn chứng minh retention có giá trị thật, phải dùng TG-ReDial: ở đó 41% mẫu có nhãn lẫn, và window cũ được nhắc lại có utility .475 so với .154 nếu không được nhắc lại. Nhưng công thức TG hiện tại đang hỏng (R@10 khoảng .028, trong khi MHIM đạt .031 và HyCoRec* .0377), nên cần sửa trước.

## 6. Lưu ý về độ tin cậy

- **Chỉ 1 seed.** Mọi Δ của r3 đều ≤ 2 std của uniform. Không kết luận gì về thứ tự giữa các cell r3 khi chưa có thêm seed.
- **Baseline uniform seed 3407 lệch cấu hình (đã xác minh).** Run `r2_uniform` seed 3407 (log `20261021-222655`) dùng `hconv_dropout: 0.3`. Mọi run khác dùng 0.2: r3, c_noprop, và ip/uniform/fixed/g_q của vòng 4–5. Riêng `log/20261021-170912` cũng dùng 0.3. Hệ quả:
  - `retention_log.md` ghi vòng 4–5 là "dropout 0.3". Ghi chú đó sai: các run này dùng 0.2.
  - Mọi Δ ở seed 3407 trong vòng 4–5 bị lệch cấu hình, vì baseline chạy 0.3 còn cell chạy 0.2. Điều này áp dụng cho ip seed 3407 (+.0058 R@1), g_q_* seed 3407, và mốc A trong file này. Các trung bình "3 seed" của uniform cũng vì vậy lẫn hai cấu hình.
  - Các Δ ghép cặp ở seed 42/123 vẫn hợp lệ, và hiệu ứng đầu bảng của ip vẫn còn ở hai seed đó (mục 3.3).
  - Với các so sánh r3, chỉ dùng mốc B (`r3_rel_uniform`). Ví dụ ΔR@1 của `r2_c_noprop` (+.0025) là so với uniform dropout 0.3, nên chưa có mốc sạch: cần chạy `r2_uniform` seed 3407 ở dropout 0.2.
- Thời gian/epoch đo khi 4 job chạy song song, nên chỉ so sánh tương đối được. Ví dụ, `r3_evidence_noprop` chậm hơn `r3_evidence` dù làm ít việc hơn.

## 7. Bước tiếp theo đề xuất

```bash
# uniform seed 3407 ở cấu hình hiện tại (dropout 0.2). --force để chạy lại dù đã có kết quả cũ (dropout 0.3)
python scripts/run_cpg.py run --cells r2_uniform --dataset hredial --seeds 3407 --gpu 0 --force
# 2 seed còn thiếu cho các cell quyết định (8 job / 4 GPU)
python scripts/run_cpg.py run --cells r2_c_noprop r3_evidence_noprop r3_rel_attn r3_rel_uniform \
  --dataset hredial --seeds 42 123 --gpu 0,1,2,3 --parallel
```
`--force` áp dụng cho mọi cell trong lệnh, nên lệnh uniform phải chạy riêng như trên. Nếu không, các cell đã có kết quả ở seed 42/123 sẽ bị chạy lại.

Kết quả sẽ trả lời ba câu hỏi:

1. Lợi ích của noprop (`r2_c_noprop`, `r3_evidence_noprop`) có ổn định qua seed không, và có bằng được ip không.
2. Relevance (`rel_attn` so với `rel_uniform`) có thật không.
3. Đánh đổi giữa đầu bảng và tail khi tắt propagation có lặp lại không.

Không cần thêm seed cho `r3_age`, `r3_evidence` và `r3_evidence_noov`, vì cả ba đã trùng `r3_rel_attn` ở seed 3407.
