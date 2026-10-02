# Đánh giá sliding-window hyperedge trên ReDial / TG-ReDial

Kết quả số chi tiết + hình: [`statistic_results/REPORT.md`](statistic_results/REPORT.md)
(sinh bằng `python analysis/window_stats.py --dataset all`).
File này là phần *diễn giải* — nối từng thống kê với quyết định thiết kế.

## Quy ước

- **Session** = 1 conversation duy nhất (gộp train+valid+test, dedup theo `conv_id`).
  ReDial: 11 348 session; TG-ReDial: 10 000 session.
- **Turn** = các utterance liên tiếp cùng role được gộp lại (đúng như
  `HReDialDataset._convert_to_id`). Đây là đơn vị mà cửa sổ trượt sẽ trượt qua.
  Nếu hyperedge dựng theo *từng utterance* thì nhân hệ số ~1.5 cho ReDial
  (18.2 utt vs 12.3 turn / session).
- 3 loại hyperedge tách riêng: `item` (field `movies`, phim gợi ý/nhắc qua `@id`),
  `entity` (field `entity` — genre/đạo diễn/diễn viên… từ entity-linking),
  `word` (token nội dung, đã bỏ stopword/dấu câu).

---

## Thống kê 1 — số turn/session: regime "turn xa" có phổ biến không?

| | ReDial (EN) | TG-ReDial (ZH) |
|---|---|---|
| mean / median turn | 12.3 / 12 | 12.9 / 12 |
| p90 / p95 / max | 17 / 19 / 83 | 16 / 16 / 16 |
| % ≥10 turn | **77.9 %** | 100 % |
| % ≥15 turn | **22.8 %** | 14.5 % |
| short<5 / medium5–10 / long>10 | 0.1 % / 33.4 % / **66.5 %** | 0 % / 7.5 % / 92.5 % |

**Kết luận:**
- ReDial **không** giống trường hợp "80 % session <10 turn". 2/3 session là *long*,
  ~23 % đạt ≥15 turn → tình huống "anchor ở turn 1, recommend ở turn 15" là một
  **phân khúc thực sự (~1/4 dataset)**, không phải đuôi hiếm. Contribution nhắm
  long-range là chính đáng, chỉ cần nói rõ nó phục vụ phần long đó.
- TG-ReDial gần như cố định độ dài (10–16 turn, std 1.66) do được dựng kịch bản.
  → rủi ro "session ngắn" **không tồn tại** ở TG-ReDial; chỉ cần bận tâm ở ReDial
  (và ở đó short<5 cũng chỉ 0.1 %).

---

## Thống kê 2 — turn-gap của lần nhắc lại liên tiếp: chọn $w$

% cặp nhắc-lại-liên-tiếp có $\Delta t \le w$:

| loại | median Δt | Δt≤2 | **Δt≤3** | Δt≤5 | Δt≤10 | #cặp |
|---|---|---|---|---|---|---|
| ReDial item | 1 | 70.7 % | 88.6 % | 96.0 % | 99.6 % | 12 317 |
| ReDial entity | 2 | 63.9 % | 75.2 % | 88.5 % | 98.3 % | 4 256 |
| ReDial word | 2 | 58.0 % | 69.7 % | 85.3 % | 98.0 % | 66 022 |
| TG entity | 1 | 72.7 % | 83.9 % | 94.7 % | 99.9 % | 28 741 |
| TG word | 2 | 69.2 % | 80.2 % | 92.5 % | 99.6 % | 715 597 |
| TG item | 2 | (2/3) | (2/3) | (3/3) | — | 3 |

**Kết luận:**
- **Chọn $w = 3$** làm mặc định (knee của cả item lẫn entity), report ablation
  $w \in \{2, 3, 5\}$. $w=3$: bắt 88.6 % recurrence của item, 75.2 % của entity (EN),
  83.9 % của entity (ZH).
- Item recurrence **chặt hơn** attribute recurrence (median 1 vs 2). Có thể cho
  `item` một $w$ nhỏ (2–3) và `entity`/`word` một $w$ lớn hơn (5) — số liệu ủng hộ
  $w$ theo từng loại hyperedge.
- **Đuôi cần global**: ở $w=3$ vẫn còn **24.8 %** cặp entity và **11.4 %** cặp item
  (EN) nằm ngoài cửa sổ. Đây là con số trích dẫn trực tiếp cho "long-term dependency
  chiếm ~1/4 và không được window xử lý".
- TG-ReDial: **item-level window vô nghĩa** (chỉ 3 cặp nhắc lại trên toàn bộ 10k
  session — seeker không bao giờ nhắc lại phim, recommender liệt kê đúng 3 phim khác
  nhau một lần). Trên TG-ReDial câu chuyện sliding-window là của `entity`/`word`.

---

## Thống kê 3 — vị trí ground-truth item: window có đủ không?

"Rec turn" = turn Recommender cuối cùng có phim; target = (các) phim ở turn đó.

| | ReDial | TG-ReDial |
|---|---|---|
| target **cold** (không xuất hiện trước rec turn) | **88.3 %** | 100 % |
| target có xuất hiện trước rec turn | 11.7 % (1 764) | 0 % |
| rec turn cách cuối session (median) | 3 turn | 1 turn |

Trong nhóm 11.7 % target *có* xuất hiện trước (ReDial):

| phân nhóm | % nhóm-xuất-hiện |
|---|---|
| **recent** (lần nhắc cuối ≤3 turn trước rec) — local window đủ | 74.6 % |
| **early-only** (nhắc 1 lần, sớm, không lặp) — **cần global** | 22.3 % |
| mid (lặp nhưng lần cuối ở xa) | 3.2 % |

Khoảng cách (turn) với nhóm có xuất hiện: `rec − first_mention` mean 3.2 / median 2 /
p90 6 / max 26.

**Kết luận:**
- ReDial: phim được recommend **hầu như luôn là phim mới** do recommender đưa ra
  (88 % cold). ⇒ sliding-window **không phải cơ chế "truy hồi target từ lịch sử"**;
  vai trò của nó là mô hình hoá **ngữ cảnh sở thích đang tiến hoá** (attributes +
  phim mà seeker thích) để recommender/KG suy ra target. Cần nói đúng vai trò này
  trong motivation, tránh bị phản biện "target đâu có trong history".
- Bằng chứng cho $H_{global}$ ở ReDial = **22.3 %** tín hiệu item tái xuất hiện chỉ
  được neo *sớm* → cửa sổ ngắn đánh rơi hoàn toàn. Cộng với đuôi 24.8 % của Thống kê 2.
- rec turn cách cuối session median 3 turn ⇒ có đàm phán/tinh chỉnh sau gợi ý đầu tiên;
  "turn cuối" ≠ "turn recommend".
- TG-ReDial: 100 % cold *do thiết kế* — toàn bộ tín hiệu sở thích nằm ở `entity`.
  Không đưa ra tuyên bố về lợi ích item-window trên TG-ReDial.
- *Caveat*: bản processed đã bỏ cờ accept/reject/seen của ReDial nên "target" là
  proxy (phim ở rec turn cuối). Con số 88 % cold đủ lớn để kết luận không đổi.

---

## Thống kê 4 — decay curve (lift đồng-xuất-hiện theo gap $g$)

`lift(g)` = (số phần tử chung trung bình giữa 2 turn cách nhau $g$) / (kỳ vọng khi
độc lập). `lift = 1` ⇔ không liên hệ.

| loại | g=1 | g=2 | g=3 | g=5 | g=10 | g=15 |
|---|---|---|---|---|---|---|
| ReDial entity | 10.9 | 4.3 | 3.7 | 2.5 | 1.3 | 0.94 |
| ReDial item | 143 | 28 | **56** | 19 | 2.6 | 1.8 |
| ReDial word | 3.6 | 2.5 | 1.8 | 1.3 | 1.1 | 0.69 |
| TG entity | 12.1 | 8.2 | 6.7 | 5.0 | 0.74 | 0.0 |
| TG word | 1.6 | 1.4 | 1.3 | 1.2 | 0.73 | 0.19 |

**Kết luận:**
- **ReDial entity**: giảm đơn điệu, mượt, cắt mức 1.0 quanh $g \approx 13$.
  Khớp $\text{lift}(g) \approx \text{lift}(1)\,e^{-(g-1)/\tau}$ trên ~8 điểm đầu cho
  **$\tau \approx 3\text{–}4$ turn** → khởi tạo tham số decay có căn cứ, trùng với knee
  ở Thống kê 2. Đây là kênh "hợp lệ" nhất để dùng decay.
- **ReDial item**: **không đơn điệu** (143 → 28 → 56 → 19), có bướu ở $g=3$ = pattern
  *callback* (recommender nhắc lại phim sau khi seeker phản hồi). Giá trị tuyệt đối
  nhiễu vì item thưa (mẫu số nhỏ) — trình bày thận trọng — nhưng hình dạng ủng hộ
  luận điểm: quan hệ item **không** tuân theo decay đơn thuần ⇒ nên tách kênh global
  riêng thay vì chỉ nới $\tau$.
- **TG entity**: giảm đơn điệu rồi **rơi vực** quanh $g \approx 9\text{–}10$ (khối
  recommendation ở cuối đổi chủ đề). Decay **không dừng (non-stationary)** theo vị trí
  trong session — một quan sát đáng nói, cùng hướng với ReDial item.
- **word**: decay rất nông, giữ ~1 rất lâu (từ vựng chủ đề bền) → $w$ cho `word` có thể
  lớn hơn, hoặc word-hyperedge nên dựa vào global nhiều hơn local.

---

## Thống kê 5 — rủi ro collapse ($w \ge |\text{session}|$)

| $w$ | ReDial overall | ReDial short<5 | ReDial med 5–10 | TG overall |
|---|---|---|---|---|
| 3 | 0.05 % | 40 % | 0 % | 0 % |
| 5 | 0.71 % | 100 % | 1.7 % | 0 % |
| 10 | 33.6 % | 100 % | 100 % | 7.5 % |

**Kết luận:**
- Với $w \le 5$, collapse **không đáng kể** (ReDial ≤0.7 %, TG 0 %). Chỉ khi $w \ge 10$
  mới thành vấn đề thật (1/3 session ReDial).
- Vẫn nên thêm guard $w' = \min(w,\ |\text{session}|-1)$ như bảo hiểm rẻ tiền, nhưng
  không cần cơ chế adaptive phức tạp cho $w \in \{3,5\}$. Nếu advisor hỏi "session
  ngắn thì sao": trả lời bằng bảng này.

---

## Tổng hợp — khuyến nghị thiết kế

1. **$w = 3$** mặc định, ablation $w \in \{2,3,5\}$. Căn cứ: knee Thống kê 2 +
   $\tau \approx 3\text{–}4$ Thống kê 4 + gần như không collapse Thống kê 5.
2. **$w$ theo từng loại hyperedge**: item $w{=}2\text{–}3$ (recurrence chặt),
   entity $w{=}5$, word $w{=}5$+ hoặc nghiêng global. Số liệu Thống kê 2 & 4 ủng hộ.
3. **Kênh global là cần thiết**, bằng chứng định lượng:
   - đuôi Thống kê 2: 24.8 % cặp entity / 11.4 % cặp item (EN) ngoài $w{=}3$;
   - Thống kê 3: 22.3 % target tái xuất hiện chỉ được neo sớm;
   - Thống kê 4: quan hệ item không đơn điệu (callback), decay non-stationary.
4. **TG-ReDial**: chạy method nhưng item-window sẽ *inert*; trình bày TG-ReDial như
   câu chuyện entity/word-window. Không claim lợi ích item-level sliding-window ở đây.
5. Guard $w' = \min(w, |\text{session}|-1)$ — thêm cho chắc, tác động ~0 ở $w \le 5$.

### Mapping sang bài viết

| Thống kê | Dùng làm | Con số chốt |
|---|---|---|
| 1 | mô tả dataset / Fig 1a | 66.5 % session ReDial là *long* (>10 turn) |
| 2 | cơ sở chọn $w$ / Fig 1b | $w{=}3$ bắt 88.6 % (item) & 75.2 % (entity) recurrence |
| 3 | bằng chứng $H_{global}$ / Fig 2 | 88 % target cold; 22.3 % nhóm-xuất-hiện là early-only |
| 4 | fit $\tau$, biện minh kiến trúc | entity $\tau\approx3\text{–}4$; item non-monotonic |
| 5 | justify guard $w'$ | collapse <0.7 % khi $w\le5$ |

### Hạn chế của phân tích

- "turn" = gộp cùng-role; đổi sang per-utterance sẽ dịch số (~×1.5 ReDial).
- `entity` chứa nhiễu entity-linking (rõ nhất: entity thời gian/ngày tháng ở TG-ReDial).
- Gộp train+valid+test.
- "target" là proxy (rec turn cuối) vì bản processed đã bỏ cờ accept/reject.
- `lift` cho `item` nhiễu do item thưa — chỉ đọc *hình dạng*, không đọc giá trị tuyệt đối.
