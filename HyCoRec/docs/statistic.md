## Chuẩn bị trước

Vì HyCoRec đã có pipeline entity-linking (ConceptNet/HowNet, `en_side`/`zh_side`), bạn tận dụng luôn output linking đó thay vì làm lại từ đầu. Với mỗi session cần trích ra tối thiểu:
- `session_id`, danh sách turn theo thứ tự
- với mỗi turn: tập entity/item được linking (tách riêng **item** — thứ được recommend — với **attribute entity** — genre/actor/director...)
- `target_item` (ground-truth được recommend/accept ở cuối session)

Chạy trên cả ReDial (EN) và TG-ReDial (ZH) riêng biệt rồi so sánh — vì đặc điểm hội thoại hai ngôn ngữ/hai dataset có thể khác nhau đáng kể (điều này tự nó cũng là một điểm thú vị để trình bày).

## Thống kê 1 — Phân bố số turn/session

**Trích xuất**: đếm số turn (utterance) mỗi session.
**Tính**: mean, median, std, percentile 25/50/75/90/95, min/max.
**Vẽ**: histogram + CDF số turn/session.
**Chỉ số phái sinh**: % session có ≥5, ≥10, ≥15, ≥20 turn.
**Dùng để**: xác định "turn 1 vs turn 15" có phải tình huống phổ biến hay chỉ là đuôi hiếm — nếu ví dụ 80% session ReDial <10 turn, bạn cần nói rõ contribution nhắm vào phần đuôi dài đó, không phải toàn bộ dataset. Đây cũng là bảng chia bucket (short <5 / medium 5–10 / long >10) dùng lại ở thống kê 5.

## Thống kê 2 — Turn-gap distribution của entity/item lặp lại (quan trọng nhất, quyết định $w$)

**Trích xuất**: với mỗi session, với mỗi entity/item xuất hiện ở ≥2 turn, lấy tất cả các cặp turn liên tiếp mà nó xuất hiện $(t_i, t_{i+1})$, tính $\Delta t = t_{i+1} - t_i$.

**Tách riêng hai loại** (rất nên làm, vì pattern khác nhau):
- Δt của **item** (thứ được đề xuất/nhắc lại) — thường item bị từ chối sẽ không lặp, item được quan tâm có thể lặp gần
- Δt của **attribute entity** (thể loại, diễn viên...) — thường persistent hơn, ít suy giảm theo khoảng cách

**Tính**: histogram Δt (1,2,3,...,≥10), median, và đặc biệt **% cặp lặp lại nằm trong Δt ≤ w** cho từng ứng viên $w \in \{1,2,3,5,10\}$.

**Dùng để**: chọn $w$ có căn cứ số liệu (không đoán) — ví dụ nếu 80% cặp lặp lại của attribute entity nằm trong Δt≤3 → w=3 hợp lý cho local; đồng thời phần còn lại (Δt>w, đuôi dài) chính là population cần cơ chế global phục vụ — bạn có thể trích dẫn con số này trực tiếp làm bằng chứng "long-term dependency chiếm X% và không được giải quyết bởi window".

## Thống kê 3 — Vị trí ground-truth item so với cuối session

**Trích xuất**: với mỗi session, tìm turn đầu tiên `target_item` được nhắc đến (nếu có nhắc trước khi recommend), và turn cuối cùng nó được nhắc lại trước turn recommend.
**Tính**:
- Khoảng cách `(turn_recommend - turn_first_mention)` — phân bố.
- % session mà `target_item` **chỉ được nhắc một lần, sớm, và không lặp lại** trước khi được recommend (đây là nhóm case bắt buộc cần global/long-term, vì window ngắn sẽ bỏ lỡ hoàn toàn).
- % session mà `target_item` được nhắc/lặp gần turn cuối (nhóm mà local window đã đủ).

**Dùng để**: đây là bằng chứng thực nghiệm mạnh nhất cho $H_{global}$ — nếu tỷ lệ nhóm "chỉ nhắc sớm, không lặp" không nhỏ (ví dụ >15-20%), bạn có số liệu cụ thể để phản biện lại câu hỏi "vậy sao không dùng window là đủ".

## Thống kê 4 — Decay curve thực nghiệm (PMI theo turn-gap)

**Trích xuất**: với mọi cặp entity $(e_i, e_j)$ đồng xuất hiện trong cùng session, gộp theo bucket $\Delta t$ (0,1,2,...,≥10).

**Tính** với mỗi bucket:
$$\text{PMI}_{\Delta t}(e_i,e_j) = \log \frac{P(e_i, e_j \mid \Delta t)}{P(e_i)\,P(e_j)}$$
hoặc đơn giản hơn: normalized co-occurrence frequency theo bucket.

**Vẽ**: đường cong PMI/co-occurrence trung bình theo $\Delta t$.

**Dùng để**: 
- Nếu đường cong giảm đơn điệu → ủng hộ decay function dạng $\exp(-\Delta t/\tau)$, và bạn fit $\tau$ trực tiếp từ đường cong này (khởi tạo tham số có căn cứ thay vì random).
- Nếu đường cong **không giảm đơn điệu** — có điểm gồ lên ở Δt lớn (do recurrence/callback) — đây chính là bằng chứng định lượng cho thấy quan hệ dài hạn *không* tuân theo decay đơn thuần, củng cố lý do phải tách kênh global riêng thay vì chỉ nới rộng $\tau$.

## Thống kê 5 — Rủi ro hyperedge collapse ở session ngắn

**Trích xuất**: dùng lại bucket độ dài session ở Thống kê 1. Với mỗi candidate $w$, tính tỷ lệ session mà $w \ge |\text{session}|$ (tức local hyperedge = global hyperedge, mất ý nghĩa phân biệt).

**Tính**: % session bị "collapse" theo từng $w$, chia theo bucket short/medium/long.

**Dùng để**: quyết định có cần $w' = \min(w, |\text{session}|)$ hay một cơ chế adaptive khác, và định lượng mức độ nghiêm trọng của rủi ro sparse-session bạn từng lo ngại — tránh advisor hỏi mà không có số trả lời.

## Tổng hợp — mapping sang bài viết

| Thống kê | Trả lời câu hỏi | Dùng làm |
|---|---|---|
| 1. Turn/session | Vấn đề có phổ biến không? | Bảng mô tả dataset / Fig 1a |
| 2. Turn-gap lặp lại | Chọn $w$ thế nào? | Cơ sở chọn hyperparameter, Fig 1b |
| 3. Vị trí ground-truth | Window có đủ không? | Bằng chứng cho $H_{global}$, Fig 2 |
| 4. Decay curve (PMI) | Decay có đúng hình dạng giả định? | Fit $\tau$, biện minh kiến trúc |
| 5. Collapse risk | Thiết kế có ổn với session ngắn? | Justify adaptive window, tránh phản biện |

Thống kê 2 và 3 là hai cái nên làm trước tiên — chúng vừa cho bạn hyperparameter, vừa là bằng chứng trực tiếp nhất để đưa vào phần motivation/Figure 1 của paper.