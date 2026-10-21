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

## Ghi chú vận hành
- Máy local 7 GB RAM: full hredial gần hết bộ nhớ; chạy tuần tự. Remote dùng `config` khác local (lr 0.0005, wd 0.1, clip 1).
- Config `r2_*` sinh bằng `python scripts/run_cpg.py gen` (không commit).
