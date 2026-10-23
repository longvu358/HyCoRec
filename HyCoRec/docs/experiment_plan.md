# Các run cần cho bài báo (theo `docs/experiment.jpg`)

**Mô hình chính** là ReSCOPE với retention learned v2, cell `r2_rescope_ip`:

- `retention: learned`, `retention_gate_mode: interp`;
- `retention_prior: decay`, `retention_gamma: 0.75`;
- `window_dedupe: true`.

Mọi biến thể khác (v1, `_i`, `r3_*` evidence, `c_noprop`, shuffle, readout-only, w/o age) **không** đưa vào bài.

**Quy ước:**

- 3 seed: 3407, 42, 123.
- ReDial: `hconv_dropout: 0.2`.
- ✅ đã có · ⚠️ phải chạy lại · ⬜ chưa chạy.

**Thứ tự chạy:** RQ2 ReDial → sửa công thức TG → RQ2 TG → RQ1 → RQ3. RQ4 chỉ là phân tích trên checkpoint của RQ2, không train thêm.

---

## RQ2: Contextual relation persistence

Bảng gồm 5 hàng × 2 dataset.

| hàng | cell | ReDial | TG |
|---|---|---|---|
| Latest relation only | `r2_latest` | ⬜ ×3 | ⬜ ×3 |
| w=1 full history | `r2_turn` | ⬜ ×3 | ⬜ ×3 |
| Uniform relational history (α=1) | `r2_uniform` | ✅ 42, 123 · ⚠️ 3407 (đang ở dropout 0.3) | ⚠️ 3407 · ⬜ 42, 123 |
| Fixed temporal decay (α = 0.75^tuổi) | `r2_ip_fixed` | ⬜ ×3 | ⬜ ×3 |
| **ReSCOPE (learned)** | `r2_rescope_ip` | ✅ ×3 | ⬜ ×3 |

Ghi chú:

- `r2_ip_fixed` dùng cùng cổng `interp` và cùng γ = 0.75 với prior của ReSCOPE. Như vậy chênh lệch giữa hai hàng chỉ còn là α có được học hay không.
- TG uniform seed 3407 hiện có (log/20261022-124627) không ghi `hconv_dropout` trong config, nên chạy ở mặc định 0. Cả run đó lẫn mọi run TG khác phải chạy lại sau khi sửa công thức TG.

**Kết quả đã có (ReDial, test):**

| cell | seed | R@1 | R@10 | R@50 | MRR@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| r2_uniform | 42 | .0309 | .1663 | .3499 | .0647 | .0883 |
| r2_uniform | 123 | .0270 | .1709 | .3510 | .0617 | .0870 |
| r2_rescope_ip | 3407 | .0324 | .1700 | .3537 | .0671 | .0910 |
| r2_rescope_ip | 42 | .0334 | .1648 | .3489 | .0665 | .0894 |
| r2_rescope_ip | 123 | .0322 | .1659 | .3604 | .0655 | .0888 |

**Lệnh (ReDial, 10 run, khoảng 3 vòng × 3.5 giờ trên 4 GPU):**

```bash
python scripts/run_cpg.py run --cells r2_latest r2_turn r2_ip_fixed --dataset hredial --seeds 3407 42 123 --gpu 0,1,2 --parallel
python scripts/run_cpg.py run --cells r2_uniform --dataset hredial --seeds 3407 --gpu 3 --force   # terminal khác
```

**Cách diễn giải RQ2** (bản ngắn để chốt với cô):

- **Bảng 5 hàng tách hai trục:**
  - *Hình thành relation*: so w=1 với Uniform (w=3).
  - *Duy trì relation*: so Latest, Uniform, Fixed decay với ReSCOPE.
- **Latest so với Uniform:** dù kết quả nghiêng về bên nào, chỉ cần ReSCOPE đứng trên cả hai thì kết luận là *cắt lịch sử hay giữ đều lịch sử đều chưa đủ, cần cơ chế quyết định relation còn ảnh hưởng bao lâu*.
- **ReSCOPE so với Uniform:** đã có. R@1 tăng khoảng +.004, MRR khoảng +.003, nhất quán qua các seed. R@10/50 tương đương Uniform, nhưng chỉ viết "tương đương" nếu bootstrap xác nhận không có ý nghĩa; ngược lại phải viết là đánh đổi.
  - Cơ chế (dẫn sang RQ4): retention chặn các relation cũ lan truyền vào biểu diễn hiện tại và dồn readout về các relation gần, nên phần đầu bảng xếp hạng sắc hơn.
- **ReSCOPE so với Fixed decay:** hai hàng dùng cùng cổng và cùng prior 0.75^tuổi, nên chênh lệch chính là đóng góp của việc *học* retention.
  - Nếu Fixed ≈ ReSCOPE, đóng góp phải chuyển thành *khung persistence cùng prior lấy từ dữ liệu*. Vì vậy nên chạy `r2_ip_fixed` sớm.
- **TG:**
  - Nếu có cùng xu hướng: TG là bằng chứng mạnh, vì hội thoại được dựng theo kịch bản và chuyển chủ đề có chủ đích.
  - Nếu không có khác biệt: viết rằng ở TG ngữ cảnh chỉ đóng góp qua entity/word (item không được nhắc lại, target cold 100%), nên retention trung tính.
- **Không viết:**
  - "cải thiện recommendation" chung chung;
  - "giữ lại relation xa quan trọng";
  - "α đánh giá độ hữu ích theo nội dung".

**TG (15 run, sau khi sửa công thức):**

```bash
python scripts/run_cpg.py run --cells r2_latest r2_turn r2_uniform r2_ip_fixed r2_rescope_ip --dataset htgredial --seeds 3407 42 123 --gpu 0,1,2,3 --parallel --force
```

---

## RQ1: Overall effectiveness (rec + generation)

| hàng | cell / nguồn | ReDial | TG |
|---|---|---|---|
| **ReSCOPE** (C+P+G, rec + conv) | `rescope_full` | ⬜ ×3 | ⬜ ×3 |
| HyCoRec chạy lại trên pipeline này | `hycorec_repro` | ✅ ×3 (remote) | ✅ ×3 (remote), chạy lại nếu đổi công thức TG |
| KBRD, KGSF, RevCore, UniCRS, MHIM, HyCoRec, HiCore/HyFairCRS | lấy số từ bài gốc, giữ bảng hiện tại | — | — |

Số liệu của `hycorec_repro` nằm trên remote. Lấy ra bằng `python scripts/run_cpg.py summarize --cells hycorec_repro`.

```bash
python scripts/run_cpg.py run --cells rescope_full --dataset hredial htgredial --seeds 3407 42 123 --gpu 0,1,2,3 --parallel
```

Có 6 run. Mỗi run gồm cả rec và conv, mất khoảng 8 giờ.

---

## RQ3: Multi-scope evidence

| hàng | cell | ReDial | TG |
|---|---|---|---|
| C only | `rq3_C` | ⬜ ×3 | ⬜ ×3 |
| C + P | `rq3_CP` | ⬜ ×3 | ⬜ ×3 |
| C + G | `rq3_CG` | ⬜ ×3 | ⬜ ×3 |
| C + P + G | `r2_rescope_ip` (dùng lại từ RQ2) | ✅ | (từ RQ2) |

Subgroup (P trên người dùng có phiên khác, G trên tail target) lấy từ dump của `analyze`, không train thêm.

```bash
python scripts/run_cpg.py run --cells rq3_C rq3_CP rq3_CG --dataset hredial htgredial --seeds 3407 42 123 --gpu 0,1,2,3 --parallel
```

Có 18 run.

---

## RQ4: Retention behavior (phân tích, không train thêm)

Dùng checkpoint `r2_rescope_ip` và `r2_uniform` của RQ2.

Hình trong bài gồm 3 panel, (a) đến (c). Case study đi kèm. Sweep siêu tham số đưa vào Appendix.

**Lưu ý chung:** α tuyệt đối của ReSCOPE đã sập về khoảng 0 (97% < .05). Readout chuẩn hoá theo tỉ lệ, nên đại lượng có tác dụng là **trọng số tương đối**. Mọi hình và chú thích đều dùng trọng số tương đối, không dùng α tuyệt đối.

| phân tích | cách làm | công cụ | trạng thái |
|---|---|---|---|
| (a) α theo tuổi relation | vẽ α(tuổi)/α(tuổi 0), trung bình 3 seed kèm khoảng tin cậy, chồng thêm đường prior 0.75^tuổi | `analyze` + `scripts/extra/ret_alpha.py` | ✅ ReDial seed 3407 · ⬜ seed 42/123, TG |
| (b) Bỏ relation α cao / α thấp / ngẫu nhiên | trên checkpoint đã train, che **25% cửa sổ** của mỗi mẫu (chỉ mẫu có ≥ 3 cửa sổ) theo α cao nhất / thấp nhất / ngẫu nhiên; báo cáo mức giảm **R@1 / MRR** | cần viết chế độ đánh giá che cửa sổ (khoảng nửa ngày, không train lại) | ⬜ chưa có code |
| (c) Gain so với uniform theo độ dài hội thoại | **ΔR@1 / ΔMRR** ghép cặp theo mẫu, chia nhóm `n_turn` ≤4 / 5–8 / 9–12 / >12, gộp 3 seed | script trên dump (`rank`, `n_turn`) | ⬜ |
| Case study | xem mục dưới | `scripts/extra/case_study.py` (chưa viết) | ⬜ |
| Sweep (Appendix) | γ của prior 0.5 / 0.75 / 0.9 cho ReSCOPE; w = 1 / 3 / 5. Mỗi điểm 1 seed, chỉ ReDial | khoảng 5 run, cần thêm cell | ⬜ |

**Thông điệp dự kiến của từng panel** (phải kiểm lại khi có số):

- **(a)** Trọng số giảm rất nhanh theo tuổi. Cửa sổ tuổi 0 nặng khoảng 10× cửa sổ tuổi ≥3, dốc hơn prior 0.75^tuổi. Viết là: *retention học được hội tụ về một mức decay gắt hơn prior thống kê*.
- **(b)** Che các cửa sổ α cao (thường là cửa sổ mới nhất) làm giảm mạnh, che α thấp gần như không ảnh hưởng, che ngẫu nhiên nằm giữa. Vì α ≈ recency, chỉ viết là α xếp hạng relation **phù hợp với** recency. Không viết "α đánh giá theo nội dung".
- **(c)** Nếu gain tăng theo độ dài hội thoại, đây là bằng chứng tốt cho retention. Nếu không, bỏ panel này. Không dùng ΔR@10, vì R@10 của ReSCOPE thấp hơn uniform.
- **Sweep:** γ có khả năng ít ảnh hưởng, vì α sập về cùng một điểm bất kể khởi tạo. Viết là mô hình *không nhạy với prior*.

```bash
python scripts/run_cpg.py analyze --cells r2_rescope_ip r2_uniform --dataset hredial --seeds 3407 42 123 --force
```

### Case study

**Đổi so với ảnh.** Ảnh đề xuất "relation xa được giữ với α cao", nhưng gần như không có mẫu như vậy (α ≤ .003 ở tuổi ≥3). Thay bằng: *sau khi hội thoại chuyển chủ đề, ReSCOPE giảm trọng số relation cũ nên xếp phim đúng cao hơn Uniform*. Đây đúng là hành vi mô hình đã học.

**Nội dung hình hoặc bảng:**

| phần | nội dung |
|---|---|
| Hội thoại | 5–8 lượt rút gọn, tô màu entity/item theo cửa sổ |
| Các cửa sổ | tuổi, các node, trọng số tương đối (α / α của cửa sổ mới nhất) |
| Kết quả | top-3 của Uniform và của ReSCOPE, kèm hạng của phim đúng ở mỗi mô hình |
| Chú thích | 1–2 câu, ví dụ: "*các cửa sổ về chủ đề trước chỉ nhận khoảng 1/10 trọng số của cửa sổ hiện tại*" |

**Tiêu chí chọn mẫu.** Ghép dump của `r2_rescope_ip` và `r2_uniform` cùng seed 42 (cặp sạch về cấu hình) theo (`conv_id`, `n_turn`, `target`), rồi lọc:

1. **Gain lớn:** hạng của phim đúng ≤ 3 ở ReSCOPE và ≥ 20 ở Uniform.
2. **Đủ dài:** có ít nhất 4 cửa sổ (`n_turn` ≥ 5).
3. **Có chuyển chủ đề:** node của các cửa sổ cũ ít trùng với cửa sổ mới nhất (Jaccard < 0.2).
4. **Phim đúng gắn với ngữ cảnh gần:** phim đúng, hoặc láng giềng KG của nó, nằm trong cửa sổ mới nhất, không nằm trong các cửa sổ cũ.
5. **Đọc được:** hội thoại rõ nghĩa, tên phim quen thuộc, không có nội dung nhạy cảm.

Script in khoảng 10 ứng viên kèm văn bản hội thoại. Chọn tay 1 mẫu cho bài chính. Có thể thêm 1 **phản ví dụ** cho Appendix (thông tin cần nằm ở lượt xa, ReSCOPE bỏ qua) để làm phần limitation.

**Dữ liệu:**

- Dump đã có `conv_id`, `n_turn`, `target`, `rank`, `top20` và `retention.windows` (dạng [tuổi, α]).
- Nội dung cửa sổ, văn bản hội thoại và tên phim chưa có trong dump. Script dựng lại từ file dataset: `context_turn_*` cộng `sliding_window_hyperedges(w=3, dedupe)`, rồi ánh xạ id sang tên.

```bash
# remote
python scripts/run_cpg.py analyze --cells r2_rescope_ip r2_uniform --dataset hredial --seeds 42 --force
# local, sau khi copy hai dump về
python scripts/extra/case_study.py hredial seed42 --top 10
```

**Không viết** "mô hình hiểu người dùng đã đổi ý". Ở đây việc giảm trọng số là do recency trùng với thời điểm chuyển chủ đề; α không phân biệt nội dung giữa các cửa sổ trong cùng một mẫu.

---

## Tổng số run cần train

| | ReDial | TG |
|---|---|---|
| RQ2 | 10 | 15 |
| RQ1 | 3 | 3 (+3 `hycorec_repro` nếu đổi công thức) |
| RQ3 | 9 | 9 |
| **Tổng** | **22** | **27–30** |

## Việc chặn / cần báo cô

1. **Công thức TG** đang cho R@10 khoảng .028, thấp hơn MHIM (.031). Cần sửa trước mọi run TG.
2. **α của ReSCOPE sập về khoảng 0** (97% < .05). Hình RQ4 nên vẽ trọng số readout tương đối (α / Σα) và mô tả đúng như vậy. Case study "relation xa được giữ với α cao" gần như không có mẫu.
3. **Lưu ý "α chỉ gate hyperedge message"** trong ảnh: ReSCOPE hiện dùng α ở cả conv lẫn readout. D_v và D_e vẫn tính từ incidence nhị phân, đúng như yêu cầu.
4. **Cross-fitting của scope G** chưa được kiểm chứng.

## Đề xuất khác

| # | đề xuất | lý do | chi phí |
|---|---|---|---|
| 1 | **Paired bootstrap trên từng mẫu test**, gộp 3 seed; đánh dấu † ở các ô có ý nghĩa thống kê. Nếu còn GPU, thêm 2 seed cho Uniform và ReSCOPE | gain nhỏ (R@1 khoảng +.004, cỡ 1–2 SE) | không train thêm |
| 2 | **Xử lý tail:** ReSCOPE làm giảm tail R@50 (.0332 so với .0370). Hoặc cho thấy ở RQ3 rằng G bù lại tail (mô hình đầy đủ so với `hycorec_repro`), hoặc ghi rõ đánh đổi | ảnh bỏ bảng long-tail, nhưng động lực của HyCoRec là giảm hiệu ứng Matthew | lấy số từ remote |
| 3 | **Thống nhất số giữa RQ1 và RQ2:** RQ2 chạy `rec_only`, RQ1 chạy cả rec lẫn conv. Ghi chú rõ, hoặc dùng số rec của `rescope_full` cho cả hai bảng | số của ReSCOPE ở hai bảng sẽ khác nhau | — |
| 4 | **Kiểm tra cấu hình trong `run_cpg.py report`:** cảnh báo khi các run trong cùng một bảng khác nhau ở bất kỳ khoá nào ngoài khoá của cell | tránh lặp lại lỗi dropout 0.3/0.2 | khoảng 30 phút code |
| 5 | **Nêu trong method rằng prior γ = 0.75 lấy từ thống kê corpus** (`docs/statistic_findings.md`) | điểm cộng cho tính mới; nối liền với RQ4 (a) | — |
| 6 | **Báo cáo chi phí:** ReSCOPE chậm hơn khoảng 25–30% mỗi epoch | chủ động nêu tốt hơn để reviewer tự hỏi | một dòng trong Appendix |
| 7 | **Tag commit dùng cho các run của bài** (ví dụ `paper-runs-v1`); với baseline lấy số từ bài gốc, ghi chú khác biệt về protocol | để số trong bài khớp với code release | — |
