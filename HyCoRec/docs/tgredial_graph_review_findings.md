# TG-ReDial (htgredial): hub-degree của KG và review-info đã build — so với ReDial

Script: `analysis/graph_review_stats.py` (`PYTHONPATH=. python analysis/graph_review_stats.py`).
Số thô: `docs/statistic_results/graph_review_stats.json`.
Bổ sung cho `statistic_findings.md` (sliding-window, đã có cả TG-ReDial) và `review_data_qc.md` (chỉ ReDial).

## 1. Degree của graph `data/edger/<ds>/*_edger.pkl`

| | ReDial item/entity | **TG item/entity** | ReDial word | **TG word** |
|---|---|---|---|---|
| số node | 64 362 | 47 183 | 8 312 | 16 719 |
| mean / median | 2.4 / 1 | **8.4 / 5** | 17.1 / 7 | 11.7 / 7 |
| p90 / p99 | 2 / 4 | 9 / 32 | 45 / 131 | 22 / 85 |
| max | 12 647 | **20 647** | 812 | **2 295** |
| node deg>100 / >1000 | 64 / 15 | **180 / 27** | 173 / 0 | 121 / 1 |
| top-1% node chiếm % cạnh | **51.8 %** | 34.8 % | 10.8 % | 15.0 % |

Hub lớn nhất:
- ReDial: người cụ thể (Tracy Middendorf 12 647, Lenny Kravitz 10 472) — hub do *diễn viên/nghệ sĩ xuất hiện nhiều*.
- TG-ReDial: **tag thể loại chung** (`影视作品` 20 647, `音乐作品` 5 994, `单曲` 5 805; word `人` 2 295). Đây là hub *ngữ nghĩa rỗng*, khác bản chất ReDial.

**Kết luận:**
- Long-tail vẫn đúng ở TG-ReDial nhưng **dày hơn ở thân** (median 5 vs 1, p99 32 vs 4) và **hub nặng hơn ở đỉnh** (max 20k, 27 node >1000 vs 15). Hệ quả cho `_before_hyperconv` (mở rộng 1-hop): chi phí mỗi sample trung bình cao hơn ReDial, và một sample dính `影视作品` kéo ~20k node. Cap/streaming đã làm cho ReDial (xem memory `hypergraph_batching_reverted`) cần kiểm tra lại cap trên TG; không nên giả định cùng ngưỡng.
- Hub ở TG là category generic → nên cân nhắc **lọc/giảm trọng số hub** (ví dụ chặn node có deg>1000 khi mở rộng) thay vì chỉ cap ngẫu nhiên: mất hub này gần như không mất thông tin, khác với việc cap hub-diễn viên ở ReDial.
- Ngược lại ReDial tập trung hơn: top-1% node giữ 51.8 % cạnh, nên cắt vài hub đã giảm cạnh nhiều; ở TG cạnh phân tán hơn (34.8 %) nên cắt hub giảm ít hơn.

## 2. `*_conv_idx_to_review_info.pkl` đã build sẵn

| | ReDial (train) | **TG-ReDial (train)** |
|---|---|---|
| số conv | 14 362 | 101 397 |
| % conv có review-info | 68.2 % | **52.7 %** |
| entity/conv (khi có) | 2.51 | 1.75 |
| số entity phân biệt | 1 329 | **11 447** |
| `selected_entityIds` nằm trong `entity2id` của chính dataset | **54.8 %** (phần còn lại lệch không gian — xem `review_data_qc.md` §3) | **100 %** |
| % entity là item (∈ `item_entity_ids`) | 9.4 % | **61.3 %** |
| token/review | 11.3 | **44.7** |

**Kết luận (đã sửa sau khi kiểm tra nội dung):**
- Số liệu thô (id nằm trong `entity2id`, 61 % là item) đúng, nhưng **không có nghĩa đây là review của item**. Kiểm tra mẫu cho thấy:
  - `selected_infoListListInt` là chuỗi token **trộn lẫn, ngắt bằng `_split_`**, ví dụ `了解 了 啊 拼命 之 ， ⋯⋯ 部 政治 _split_ 音乐 ！ 亲情 …` — giống mảnh hội thoại/ngữ cảnh, không phải câu review phim.
  - Nội dung **thay đổi theo từng conv** cho cùng một entity (chỉ 1 935/11 447 entity có info giống hệt nhau giữa các conv, dù 9 512 entity xuất hiện ở nhiều conv) → là thứ gắn với *conv*, không phải với *item*.
  - Entity trong đó gồm cả diễn viên (`艾玛·罗伯茨`) lẫn phim; độ dài list luôn bằng số entity (0–3), tức cặp (entity, đoạn token) theo conv.
- Hệ quả: **không thể dựng `E_top^+(R_i)` (map item → entity tích cực trong review) từ file này.** Không có corpus review tiếng Trung per-item; kết luận cũ "Nhánh 2 khả thi hơn ở TG" **không đứng vững**, và `use_review_hypergraph: false` trong `tgredial.yaml` vẫn là cấu hình đúng.
- Nguồn gốc/ngữ nghĩa thật của file này (rất có thể là tín hiệu "selected" kiểu MHIM từ các conv tương tự) **chưa xác minh**. Ở ReDial cũng chưa giải mã được (`review_data_qc.md` §4).
- Cảnh báo: **ReDial** `train` và `test` cho số liệu giống hệt nhau (14 362 conv, 9 797 non-empty, 1 329 entity phân biệt) → nhiều khả năng `test_conv_idx_to_review_info.pkl` là bản sao của train. TG thì ba split khác nhau.

## 3. Ghép với `statistic_findings.md`

TG-ReDial: item-window inert (100 % target cold), entity/word là kênh chính, entity decay rơi vực ở g≈9–10. Kênh review (Nhánh 2) **chưa có dữ liệu** cho TG; muốn làm cần một corpus review tiếng Trung per-item (ví dụ Douban) bên ngoài.

## Hạn chế
- Hub-degree đo trên edger đã build, không phải trên graph per-sample (chưa đo kích thước subgraph sau 1-hop).
- Chưa xác định ngữ nghĩa thật của `*_conv_idx_to_review_info.pkl`.
