# Kịch bản thực nghiệm CPG-Hypergraph

Oct 16, 2026 · @Vu Hoang Long

## Tổng quan

Phần thực nghiệm gồm 6 RQ theo đúng thứ tự của HyCoRec, HiCore và HyFairCRS; mỗi RQ chỉ có **một bảng hoặc một hình**. Mọi số trong các bảng/hình mẫu bên dưới là **số giả**, chỉ để duyệt cách trình bày.

| RQ | Câu hỏi | Thể hiện | Cần chạy |
| --- | --- | --- | --- |
| — | Dữ liệu | Bảng thống kê | Chỉ đếm |
| RQ1 | Gợi ý tốt hơn baseline không? | Bảng R/M/N@10, @50 | Mô hình đầy đủ, 3 seed |
| RQ2 | Hội thoại có giữ chất lượng không? | Bảng Dist-2/3/4 | Mô hình đầy đủ |
| RQ3 | Có giảm hiệu ứng Matthew không? | Bảng coverage C@k + LT@10 | Mô hình đầy đủ |
| RQ4 | Mỗi thành phần đóng góp bao nhiêu? | Bảng "w/o X" | 7 biến thể, 3 seed |
| RQ5 | Siêu tham số ảnh hưởng thế nào? | Một hình, 4 panel | Lưới d, L, w, k\_asp |
| RQ6 | Một ví dụ cụ thể | Bảng top-5 | Chọn từ output |

Số của baseline lấy từ paper gốc, như cả bốn paper vẫn làm; bảng chỉ báo trung bình, không có ±std.

## Đối chiếu với phần thực nghiệm của các paper trước

Cả bốn paper cùng một khung: bảng gợi ý, bảng hội thoại, bảng/hình hiệu ứng Matthew, ablation dạng "w/o X", phân tích siêu tham số. Không paper nào báo std, đo chi phí bằng số, hay ablate phần fusion.

|  | MHIM (2023) | HyCoRec (ACL 2024) | HiCore (EMNLP 2024) | HyFairCRS (Findings ACL 2025) |
| --- | --- | --- | --- | --- |
| Dữ liệu | ReDial, TG-ReDial (chia lại theo user, 8:1:1) | ReDial, TG-ReDial (review từ IMDb, Douban) | ReDial, TG-ReDial + OpenDialKG, DuRecDial | Như HiCore |
| Metric gợi ý | R/M/N@10, @50 | R/M/N@10, @50 | R/M/N@10, @50 | R/M/N@10, @50 |
| Metric hội thoại | Dist-2/3/4 | Dist-2/3/4 | Dist-2/3/4 | Dist-2/3, BLEU-2/3 |
| Hiệu ứng Matthew | Không có | C@5/10/15/20, Iso-Index | C@k, A@K, L@K (long-tail ratio) | A@K, Gini@K, KL@K, D@K (K = 5–20) |
| Ablation | w/o Contrast, Session, Knowledge, HyperConv, Extension (R@10/50) | w/o item/entity/word hypergraph, w/o item reviews (R@10/50) | w/o từng hypergraph trong 9 cái (R@10/50) | w/o G\_e/G\_i/G\_w/G\_r, w/o line graph (K = 5, 10) |
| Siêu tham số | Số node quan trọng | d, số lớp HConv, số lớp pooling | d, β, số lớp, ngưỡng siêu cạnh | Số lớp, d |
| Case study | Không | Có (đa dạng thể loại) | Không | Không |

**Đã chỉnh theo quy ước này:** bảng chính đổi sang R/M/N@10, @50 với đủ nhóm baseline; thêm bảng hội thoại; ablation chính đổi sang dạng "w/o X" bỏ từ mô hình đầy đủ; thêm bảng Matthew dùng đúng các metric trên. Các phân tích không có trong paper trước được tách ra khỏi bộ chính (xem mục cuối).

Nguồn: [MHIM](https://arxiv.org/abs/2305.04798), [HyCoRec](https://aclanthology.org/2024.acl-long.138), [HiCore](https://aclanthology.org/2024.emnlp-main.86), [HyFairCRS](https://arxiv.org/abs/2507.02000).

## Dữ liệu

Bảng thống kê như MHIM Table 1, thêm một dòng tỉ lệ item có review (vì phương pháp dùng review).

&#91;image: Bảng dữ liệu (số giả)\]

## RQ1. Kết quả gợi ý

Bảng giống Table 1 của HyCoRec, HiCore, HyFairCRS: 6 metric × 2 bộ dữ liệu, baseline chia nhóm, tốt nhất in đậm, nhì gạch dưới, \* là có ý nghĩa thống kê.

&#91;image: Bảng RQ1 (số giả)\]

**Kết luận cần có:** CPG-Hypergraph tốt nhất trên cả hai bộ dữ liệu. Cần chạy thêm HyCoRec trên pipeline của mình để chắc rằng pipeline tái hiện đúng số báo cáo.

## RQ2. Kết quả hội thoại

Bảng giống Table 2 của HyCoRec: Dist-2, Dist-3, Dist-4.

&#91;image: Bảng RQ2 (số giả)\]

**Kết luận cần có:** phương pháp chỉ đổi biểu diễn người dùng, nên chất lượng hội thoại giữ được hoặc tốt hơn chút.

## RQ3. Hiệu ứng Matthew

Bảng giống Table 3 của HyCoRec (Coverage@5/10/15/20), thêm cột LT@10 (tỉ lệ item đuôi dài trong top-10) như HiCore Table 4. Hai định nghĩa cần ghi trong paper vì các paper trước không ghi:

- C@k: tỉ lệ item xuất hiện trong ít nhất một danh sách top-k trên tập test.
- LT@10: tỉ lệ vị trí top-10 thuộc item đuôi dài (item ngoài nhóm đầu chiếm 80% lượt nhắc trong train).

&#91;image: Bảng RQ3 (số giả)\]

**Kết luận cần có:** coverage và LT@10 cao hơn các baseline, nhờ siêu cạnh review/aspect phủ cả item ít được nhắc.

## RQ4. Ablation

Bảng giống HyCoRec Table 4 và HiCore Table 5: dòng đầu là mô hình đầy đủ, mỗi dòng sau bỏ đi một thành phần; chỉ R@10 và R@50.

&#91;image: Bảng RQ4 (số giả)\]

Câu hỏi "G khác gì phần mở rộng của MHIM" được trả lời ngay trong bảng bằng dòng *G → MHIM extension* (thay G bằng cách truy xuất của MHIM), không cần mục riêng.

**Kết luận cần có:** bỏ thành phần nào cũng giảm; thay G bằng MHIM extension cũng giảm.

## RQ5. Siêu tham số

Một hình như HyCoRec Figure 2: 4 panel (chiều embedding d, số lớp HConv L, window w, kích thước aspect k\_asp), trục y là R@10, mỗi bộ dữ liệu một hàng.

&#91;image: Hình RQ5 (số giả)\]

**Kết luận cần có:** chọn giá trị ở đỉnh, mô hình không quá nhạy quanh giá trị đó.

## RQ6. Case study

Một bảng: đoạn hội thoại, rồi top-5 của HyCoRec và của CPG-Hypergraph. Item đích in đậm, item đuôi dài có dấu †. Cùng ý với HyCoRec Figure 3 (cho thấy gợi ý đa dạng hơn) nhưng gọn hơn.

&#91;image: Bảng RQ6 (ví dụ minh hoạ)\]

**Chọn ca:** item đích là item đuôi dài, HyCoRec không đưa vào top-5 còn CPG-Hypergraph có.

## Phân tích phụ (không vào bài chính)

Các phân tích trước đây (chẩn đoán phía ứng viên Eq. 17, chi phí so với MHIM, attention theo lượt, theo độ dài lịch sử, thiết kế fusion, độ chồng lấn giữa scope) được tách khỏi bộ RQ. Code vẫn giữ trong thư mục `extra/` để dùng nội bộ hoặc khi reviewer hỏi. Riêng chẩn đoán Eq. 17 nên chạy trước khi chốt dòng "w/o candidate enrichment" trong RQ4.

## Câu hỏi mở cần thầy góp ý

- [ ] **Bộ dữ liệu xuyên miền.** HiCore và HyFairCRS thêm OpenDialKG và DuRecDial. Scope Personal cần nhiều phiên của cùng một người dùng; hai bộ này có thể không nhóm hội thoại theo user. Có nên bổ sung, hay giữ hai bộ và nêu ở Limitations?
- [ ] **Số baseline lấy từ paper.** Bảng RQ1–RQ3 lấy số baseline từ paper gốc như các paper trước. Thầy có đồng ý không?
- [ ] **Metric Matthew.** RQ3 dùng C@k của HyCoRec cộng LT@10 của HiCore. Có cần thêm Gini của HyFairCRS không?
- [ ] **Phân tích phụ.** Có phân tích nào trong mục phụ nên đưa lại vào bài chính không (ví dụ chi phí so với MHIM)?
