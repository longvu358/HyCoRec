# ReSCOPE retention — nhật ký thực nghiệm

Theo dõi từng vòng: thay đổi, kiểm tra, kết quả. Thiết kế gốc: `docs/retention.md`. Code: `cpc.py` (`RetentionGate`),
`hycorec.py` (`_context_retention`, `CustomHypergraphConv.edge_gate`). Cell: `scripts/cpg_manifest.py` (`r2_*`).
Phân tích α: `python scripts/extra/ret_alpha.py <dump.jsonl>`. Mọi con số dưới đây là **một seed (3407), hredial**.

## Vòng 1 — gate attenuating (`r2_rescope`, commit 5335f6e)
- Thiết kế: α = σ(MLP([h; h*q; emb(tuổi)]) + 3), dùng chung 3 field; message `α * m_e`, D nhị phân; readout trọng số `max α`.
- Đẳng thức: `retention: none` khớp HEAD (sai lệch 1.2e-7); γ=1 khớp uniform (1.2e-7); α khởi tạo = σ(3) = 0.9526.
- Kết quả (remote, 10 epoch, rec): test R@1 0.031, R@10 0.1707, R@50 0.3493, tail R@10 0.0057; valid R@10 0.1904.
- **α sau huấn luyện** (4772 mẫu, 25477 window): mean 0.989, 99.5% > 0.9, 0% < 0.5; theo tuổi phẳng
  (tuổi 0: 0.9945, tuổi 12–15: ~0.983, corr(tuổi, α) = −0.08); độ lệch chuẩn trong mẫu 0.0076 < giữa các mẫu 0.0117;
  `q_age > 0`: 0%.
- Kết luận: gate **không học quên**, mở dần về 1 (ngược hướng weight decay 0.1).

### Chẩn đoán (giả thuyết, chưa kiểm chứng bằng run đối chứng)
| # | Vấn đề | Mức |
|---|---|---|
| 1 | Gate chỉ làm giảm tín hiệu → lẫn với thang đo hàng C; α chặn ở 1 nên loss kéo về biên | cao |
| 2 | Gradient tới α gián tiếp (1 trong 7 key MHA); target cold 88% (ReDial) / 100% (TG) | cao |
| 3 | q luôn là window mới nhất (`q_age`=0) → thông tin thấp | trung bình |
| 4 | Khởi tạo gần biên bão hoà (σ'≈0.045) | thấp |
| 5 | Không có prior từ thống kê (τ≈3–4 turn, γ≈0.75) | trung bình |
| 6 | Chưa có phân tích utility (α vs mức giảm điểm item đích) | cần cho paper |

Đã bỏ qua bước so sánh `r2_uniform`/`r2_fixed`/`r2_shuffle` theo yêu cầu; **chưa biết** retention giúp hay hại về R@10.

## Vòng 2 — nội suy về node + prior decay (code xong, chưa có kết quả)
- (A) `retention_gate_mode: interp`: `out_v = D⁻¹ Σ_e [α_e m_e + (1−α_e) x_v]`. α=1 ≡ conv gốc; α=0 → node giữ
  embedding của chính nó; độ lớn đầu ra không phụ thuộc α. Test: `test_edge_gate_interp_identity_and_self_fallback`.
- (B) `retention_prior: decay` (γ=0.75): α khởi tạo `0.75^age` (kẹp 0.02–0.98); weight decay kéo về prior này.
  Test: `test_retention_gate_decay_prior_starts_at_gamma_pow_age`.
- Cell mới: `r2_rescope_i` (A), `r2_rescope_ip` (A+B). `r2_rescope` giữ nguyên nghĩa vòng 1 (gate_mode scale).
- Kiểm tra đã làm: pytest 23 pass / 1 fail cũ (`test_rows7_matches_eq13_mean_pool`, fail từ trước); smoke 1% dữ liệu
  `r2_rescope_ip` chạy xong 1.8 phút, không Traceback/NaN. **Chưa** so lại `none` với HEAD ở mức model sau thay đổi
  này (chỉ có test đơn vị α=1); chưa có run đủ dữ liệu.

### Kết quả vòng 2 (hredial, seed 3407, rec-only, log/20261021-170912)
| cell | valid R@10 (tốt nhất) | test R@10 | α mean | α<0.05 | corr(age,α) |
|---|---|---|---|---|---|
| r2_rescope (v1 scale) | 0.1904 | 0.1665 | 0.989 | ~0% | ≈0 (phẳng) |
| r2_rescope_i (interp) | 0.2000 (ep5) | 0.1667 | 0.029 | 88% | -0.13 |
| r2_rescope_ip (interp+decay) | 0.2015 (ep5) | 0.1671 | 0.011 | 96% | -0.35 |
| full (không retention, 4 seed) | — | 0.168–0.173 | — | — | — |

Quan sát:
- **α sập về ≈0**, ngược vòng 1 (≈1). Với `interp`, α→0 nghĩa là hàng C không còn propagation qua hyperedge (node giữ embedding thô, pooled trung bình vẫn còn): gate học cách *tắt conv của scope C*, không học cách quên chọn lọc. `ip` khởi tạo ở 0.75^age nhưng cả age 0 cũng về 0.036.
- Phương sai trong mẫu (0.014–0.031) nhỏ hơn giữa các mẫu (0.015–0.045) → α vẫn là hệ số thang theo mẫu, không phải bộ chọn window. `q_age>0` = 0% (q luôn là window mới nhất).
- Test R@10 của 3 biến thể retention (0.1665/0.1667/0.1671) gần như trùng nhau và thấp hơn `full` (~0.172) khoảng 0.005. Chưa kết luận được: `full` là run joint, chưa có đối chứng cùng ngân sách (`r2_uniform`).
- Khoảng cách valid→test lớn (0.20 → 0.167); rec_loss train giảm 7.9→4.3 còn valid loss tăng từ epoch 3: overfit mạnh, điều này giải thích vì sao tắt hàng C có thể được thưởng.
- Cảnh báo: gap valid→test của các run retention (+0.03) lớn hơn hẳn `full` (+0.008); chưa rõ do retention, `window_dedupe` hay `rec_only` (cần `r2_uniform` để tách).
- `alpha_f` trong log là vết của `ScopeFusion` không dùng khi `fusion_mode: rows7` (không phải lỗi).
- Chưa dùng được làm bằng chứng cho claim "learned retention" trong abstract.

Diễn giải tạm: end-to-end loss không có tín hiệu để gate ưu tiên window gần; nó chỉ thấy C là nguồn overfit. Gate hiện tại ≠ retention theo nghĩa của paper.

### Cần làm tiếp
1. (Đã xong, xem kết quả trên) chạy `r2_rescope_i` và `r2_rescope_ip` (seed 3407, rec, 10 epoch), `analyze`, rồi `ret_alpha.py`.
   Câu hỏi: α có còn ≈ 1 và phẳng theo tuổi không? Với `ip`, α có rời khỏi `0.75^age` không (nếu bám sát prior thì gate chỉ là decay cố định)?
2. Nếu α vẫn phẳng/≈1: cân nhắc giám sát trực tiếp (C: BCE phụ với nhãn utility = giao với láng giềng KG của item đích; target chỉ vào loss).
3. Viết phân tích utility (leave-one-edge-out, riêng tập target warm 11.7%).
4. Kiểm cross-fitting của scope G trước khi chốt số RQ1.
5. Nếu cần so sánh: `r2_uniform`, `r2_fixed_g07`, `r2_latest`, `r2_shuffle` cùng seed (đang bỏ qua).

## Vòng 3 — đối chứng uniform vs fixed (hredial, seed 3407, rec-only, log/20261021-222655)
Lưu ý: cả hai run dùng `hconv_dropout: 0.3` (commit 218bae2); các run vòng 2 ở local dùng 0.2, nên không so trực tiếp với vòng 2.

| cell | ep chọn | valid R@10 / R@50 | test R@1 | test R@10 | test R@50 | MRR@10 | NDCG@10 | tail R@10 / R@50 |
|---|---|---|---|---|---|---|---|---|
| r2_uniform (α=1) | 5 | .1977 / .4226 | .0265 | .1692 | .3537 | .0624 | .0872 | .0057 / .0412 |
| r2_fixed_g07 (α=0.7^age, scale, both) | 4 | .1907 / .4180 | .0272 | .1688 | .3549 | .0624 | .0871 | .0067 / .0373 |

- Không phân biệt được: chênh test ≤ .0012 ở mọi metric chính, đổi dấu giữa R@10 và R@50. Đường cong valid theo epoch trùng nhau ±.004. Fixed bị chọn ep4 chỉ vì R@50 ep5 hoà ep4 (.4180).
- Gap valid→test của uniform (+.029) cũng lớn như của các run retention, nên **gap không do retention** mà do cấu hình chung của các cell r2 (`rec_only` / `window_dedupe` / `hconv_dropout`).
- Phép thử tuyến tính không cần train (proxy wᵀÂ^G) cho cùng kết luận: decay trên phần ngữ cảnh chỉ thêm khoảng +0.3–1 điểm R@10. Trong khi đó, truy vấn G (`readout + q_p`) gồm khoảng 93% node history; trộn history với trọng số 1 làm proxy R@10 giảm từ .219 xuống .102 (ReDial) và từ .080 xuống .001 (TG).
- Hệ quả: retention ở scope C hiện không tạo chênh lệch đo được. Nên ưu tiên sửa cách tạo truy vấn G (tách context và history) trước khi đưa α sang G.

## Vòng 4 — thành phần truy vấn G (`g_query`, extra-S13) + `r2_rescope_ip` với dropout 0.3 (hredial, log/20261022-021023, -021049)
Tất cả dùng `hconv_dropout: 0.3`, `window_dedupe`, rec-only. Baseline `r2_uniform` có 3 seed (3407 lấy từ log/20261021-222655).
Δ là chênh lệch ghép cặp theo seed so với `r2_uniform`.

| cell | n seed | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | ΔR@10 theo seed (42 / 123 / 3407) |
|---|---|---|---|---|---|---|---|
| r2_uniform (`ctx+hist`) | 3 | .0281±.0024 | .1688±.0023 | .3515±.0020 | .0629±.0016 | .0875±.0007 | — |
| g_q_ctx | 3 | .0279±.0024 | .1693±.0043 | .3507±.0008 | .0622±.0009 | .0870±.0002 | −.0019 / +.0006 / +.0029 |
| g_q_split | 3 | .0292±.0012 | .1682±.0032 | .3526±.0013 | .0634±.0014 | .0877±.0016 | −.0017 / −.0013 / +.0013 |
| g_q_hist | 1 | .0267 | .1613 | .3472 | .0615 | .0848 | – / – / −.0079 |
| r2_rescope_ip (dropout 0.3) | 1 | .0324 | .1700 | .3537 | .0671 | .0910 | – / – / +.0008 |

- **`ctx` và `split` ≈ `ctx+hist`**: mọi Δ trung bình ≤ .0011, nhỏ hơn std giữa các seed. Phép thử tuyến tính (history làm loãng truy vấn G,
  R@10 proxy giảm từ .219 xuống .102) **không chuyển thành kết quả trên mô hình đã train**: mô hình đã bù được, nên phần history trong r^G là thừa chứ không có hại.
- **`hist` (bỏ ngữ cảnh khỏi truy vấn G) thấp hơn khoảng .008 R@10** (khoảng 3.4 std của uniform, 1 seed): phần ngữ cảnh của r^G là phần có ích; phần history trùng với P.
- Hệ quả: dùng `g_query: ctx` được như một cách đơn giản hoá (truy vấn nhỏ hơn, khoảng 19.9 phút/epoch so với 21.4), nhưng không phải nguồn cải thiện. Đưa retention vào G không còn lý do.
- **`r2_rescope_ip` ở dropout 0.3** cho kết quả đầu bảng tốt hơn rõ: R@1 +.0058, MRR@10 +.0047, NDCG@10 +.0038 so với uniform seed 3407. R@1 .0324 và MRR .0671
  cao hơn cả 9 run còn lại, nhưng R@10/R@50 không đổi. Mới có 1 seed; vòng 2 (dropout 0.2) không thấy hiệu ứng này. Chi phí khoảng 27.6 phút/epoch (+30%).
  Cần thêm seed 42/123 và dump α trước khi kết luận.

## Vòng 5 — thêm seed cho fixed/ip (hredial, log/20261022-142619) + TG `g_query` (log/20261022-124627)
**hredial, 3 seed (3407 / 42 / 123), dropout 0.3, Δ ghép cặp so với `r2_uniform`:**

| cell | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 | tail R@50 | phút/epoch |
|---|---|---|---|---|---|---|---|
| r2_uniform | .0281±.0024 | .1688±.0023 | .3515±.0020 | .0629±.0016 | .0875±.0007 | .0370 | ~21.4 |
| r2_fixed_g07 | .0285 (+.0003) | .1699 (+.0011) | .3541 (**+.0026**) | .0632 (+.0003) | .0880 (+.0004) | .0380 | ~23 |
| r2_rescope_ip | **.0327±.0007 (+.0045)** | .1669 (−.0019) | .3543 (+.0028) | **.0663 (+.0034)** | **.0897 (+.0022)** | .0332 (−.0038) | ~27 |

- ip: ΔR@1 theo seed +.0025 / +.0052 / +.0058 và ΔMRR@10 +.0017 / +.0038 / +.0047, **dương ở cả 3 seed** (t ghép cặp ≈ 4.4, df=2).
  Hiệu ứng tập trung ở đầu bảng xếp hạng; ΔR@10 theo seed lẫn dấu (+.0008 / −.0015 / −.0050), tail R@50 giảm. R@1 có std .0007, ổn định hơn hẳn uniform (.0024).
- fixed γ=0.7: ΔR@50 +.0012 / +.0040 / +.0025, dương ở cả 3 seed nhưng nhỏ (≈ 1 std); đầu bảng không đổi. Vậy **decay cố định không tạo ra hiệu ứng đầu bảng của ip**.
- Chưa tách được nguyên nhân: ip khác fixed ở **cả** cổng nội suy (`interp`), α học được, lẫn prior decay. Vòng 2 cho thấy α của ip sập về khoảng 0, tức scope C gần như
  tắt propagation qua hyperedge. Giả thuyết cạnh tranh: lợi ích đến từ "C không propagate" chứ không phải từ retention. Cần: dump α của ip ở dropout 0.3,
  và đối chứng `interp` với α cố định bằng 0 (chưa có cell).

**TG-ReDial, seed 3407:** uniform R@10 .0277 / R@50 .0701; g_q_ctx .0284 / .0708; g_q_split .0274 / .0701; g_q_hist .0277 / .0701 (gần như trùng với uniform ở từng epoch).
- Mọi Δ ≤ .0007, trong khi std giữa các seed của `full` TG là khoảng .003 R@10. Việc g_q_hist trùng uniform đến 4 chữ số cho thấy ở TG, hàng G phía user gần như không được dùng.
- Số TG ở công thức này (.020–.028) thấp hơn MHIM (.031) và HyCoRec* (.0377); valid rec_loss tăng 10.6 → 18.7. Công thức TG cần xử lý riêng trước khi so các biến thể nhỏ.

### Dump α của `r2_rescope_ip`, dropout 0.3, seed 3407 (ckpt 20261022-021023, `analyze --force`)
- α mean .0094, std .0214; 97.2% < .05; trung vị .0027. Theo tuổi: tuổi 0 .030, tuổi 1 .006, tuổi 2 .012, tuổi ≥3 ≤ .003; corr(tuổi, α) −.33.
  Std trong mẫu .012 so với std giữa các mẫu .013. Vẫn sập giống vòng 2 (dump trước đó là file của dropout 0.2).
- Hệ quả: ở propagation, C của ip gần như không propagate (≥97% message bị tắt, node giữ lin(x)+b).
  Ở readout, trọng số max-α được chuẩn hoá `/Σw`, nên **tỉ lệ** mới là cái có tác dụng: node ở window tuổi 0 nặng khoảng 10× node tuổi ≥3,
  tức readout C của ip là một trọng số recency rất gắt dù α tuyệt đối gần 0.
- Vậy ip = (C không propagate) + (readout C nghiêng mạnh về turn gần nhất). `r2_c_noprop` chỉ tách được thành phần thứ nhất
  (readout của nó không gán trọng số). Nếu noprop < ip, thành phần thứ hai cần thêm một đối chứng cố định: α=0 ở conv + readout γ^tuổi.

## Vòng 6 — thiết kế lại cơ chế: retention dựa trên bằng chứng (`retention: evidence`, cell `r3_*`)
### Vì sao cơ chế cũ không hợp lý (lỗi cấu trúc, không chỉ số liệu)
- α được tính lại từ đầu ở mỗi turn, chỉ từ nội dung window và tuổi; nó không biết gì về **các turn đến sau** window (nhắc lại, chuyển chủ đề),
  nên chỉ có thể là một hàm của độ gần.
- α gate conv trên các window chồng lấn (w=3): tác dụng của nó khó xác định, và nó phải gánh luôn việc "conv ở C có hại", nên sập về 0.
- Readout dùng max α trên các window chứa node, tức α của window trẻ nhất; retention bị thu về tuổi của node, mất cấu trúc relation.
- σ độc lập rồi chuẩn hoá /Σw: không có mốc, nên α trôi thành hệ số thang theo mẫu.

### Phân tích tiền nghiệm (không train): `scripts/extra/evidence_probe.py <ds>`, fit trên valid, đánh giá trên test
Nhãn utility u_e = target ∈ e hoặc nối 1-hop với node của e trong Â^G. Đo bằng AUC ghép cặp **trong cùng mẫu**, tức đúng loại tín hiệu
mà softmax trên các window khai thác được.

| scorer | hredial | htgredial |
|---|---|---|
| recency (−tuổi) | .653 | .748 |
| số lần nhắc lại | .605 | .735 |
| rel_now (liên quan đến turn hiện tại) | **.747** | **.779** |
| LR tuổi + kích thước + có item | .706 | .769 |
| LR toàn bộ bằng chứng | **.786** | **.795** |
| LR bằng chứng, bỏ rel_now | .730 | .771 |
| số mẫu có nhãn lẫn (nơi việc chọn window có tác dụng) | 550/3118 (18%) | 1154/2833 (41%) |

- Tín hiệu vượt quá tuổi là có thật. Thành phần mạnh nhất là **liên quan đến turn hiện tại** (→ số hạng relevance với q = u_T).
- **Nhắc lại làm relation không phai** (→ forget gate có đặc trưng overlap): ở TG, P(u) của window tuổi ≥7 là .154 nếu không được nhắc lại, .475 nếu được nhắc lại
  (cao ngang window tuổi 1 không được nhắc lại, .396). ReDial cùng chiều nhưng yếu hơn.
- Role (seeker_share) yếu. Trần lợi ích ở ReDial nhỏ (chỉ 18% mẫu có nhãn lẫn); TG có không gian lớn hơn.
- Cảnh báo: nhãn dùng chung Â^G với rel_now/cont_later nên AUC có thể bị thổi phồng; nhãn 1-hop khá lỏng (P(u) = .61 ở ReDial).
  Bài học từ vòng 4: proxy chỉ chỉ ra hướng, không đo được độ lớn.

### Cơ chế mới (cpc.EvidenceRetention)
- Đơn vị là relation: r_e^f = mean các node của e sau conv; conv ở C **không bị gate** (`c_propagate: false` thì C chỉ còn lin(x)+b).
- f_{e,t} = σ(MLP([h_e; u_t; h_e⊙u_t; ov(e,t)]) − 4) cho mỗi turn t > t_e; log ρ_e = Σ log(1 − f_{e,t}). Window mới nhất có ρ = 1 (mốc).
- β_e = softmax_e((W_q u_T)·(W_k h_e)/√d + log ρ_e); c^C_f = Σ_e β_e r_e^f. Khởi tạo trung tính: W_k = 0, mỗi turn quên khoảng .018.
- Thang ablation (mỗi cell bỏ đúng một thành phần): `r3_rel_uniform` (mean trên relation) → `r3_rel_fixed` (ρ = .7^tuổi) → `r3_rel_attn`
  (chỉ relevance) → `r3_age` (relevance + tốc độ quên học được) → **`r3_evidence`**; thêm `r3_evidence_noov`, `r3_evidence_noprop`.
- Kiểm tra: test `test_evidence_retention_special_cases` (none = mean, const = γ^tuổi, mốc ρ=1, khởi tạo trung tính) và
  `test_evidence_retention_depends_only_on_later_turns`; gradient tới cả hai nhánh sau bước đầu; smoke 1% dữ liệu chạy được (khoảng 2.7 phút, so với 1.9 của uniform).

## Ghi chú vận hành
- Máy local 7 GB RAM: full hredial gần hết bộ nhớ; chạy tuần tự. Remote dùng `config` khác local (lr 0.0005, wd 0.1, clip 1).
- Config `r2_*` sinh bằng `python scripts/run_cpg.py gen` (không commit).
