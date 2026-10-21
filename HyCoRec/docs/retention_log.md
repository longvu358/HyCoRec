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

### Cần làm tiếp
1. Trên remote: chạy `r2_rescope_i` và `r2_rescope_ip` (seed 3407, rec, 10 epoch), `analyze`, rồi `ret_alpha.py`.
   Câu hỏi: α có còn ≈ 1 và phẳng theo tuổi không? Với `ip`, α có rời khỏi `0.75^age` không (nếu bám sát prior thì gate chỉ là decay cố định)?
2. Nếu α vẫn phẳng/≈1: cân nhắc giám sát trực tiếp (C: BCE phụ với nhãn utility = giao với láng giềng KG của item đích; target chỉ vào loss).
3. Viết phân tích utility (leave-one-edge-out, riêng tập target warm 11.7%).
4. Kiểm cross-fitting của scope G trước khi chốt số RQ1.
5. Nếu cần so sánh: `r2_uniform`, `r2_fixed_g07`, `r2_latest`, `r2_shuffle` cùng seed (đang bỏ qua).

## Ghi chú vận hành
- Máy local 7 GB RAM: full hredial gần hết bộ nhớ; chạy tuần tự. Remote dùng `config` khác local (lr 0.0005, wd 0.1, clip 1).
- Config `r2_*` sinh bằng `python scripts/run_cpg.py gen` (không commit).
