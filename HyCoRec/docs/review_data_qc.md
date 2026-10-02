# QC dữ liệu review RevCore + đối chiếu với dữ liệu HyCoRec đã build

Script: `analysis/review_data_check.py` (`python analysis/review_data_check.py -d hredial`).
Kết quả thô: `docs/statistic_results/review_data_check.json`.

## 1. Chất lượng corpus RevCore tự thân

`movieid2review_dict.pkl` (`data/reviews/raw/revcore/`, xem `PROVENANCE.md`):

- 2711 phim, 18 phim rỗng (không review), 369 phim có review trùng lặp nội bộ (cần dedupe khi build).
- Số review/phim: min 0, max 100, **median 1**, mean 25.9 — lệch dài đuôi (đa số phim chỉ có 1 review, một số phim rất nhiều).
- Độ dài review: min 1 từ, max 1869, **median 154 từ**, p10 = 44 từ — đa số là review thật, nhiều câu, không phải đoạn rời rạc. Có vài entry rác (`"SPOILERS"`, `"0.01/10"`, < 5 từ) — cần lọc ngưỡng tối thiểu khi build.
- Non-ASCII bất thường: 12/70,290 review (0.017%) — sạch, hầu như thuần tiếng Anh, không lỗi encoding.
- Spot-check thủ công 3 phim ngẫu nhiên (Kubo and the Two Strings, A Girl Like Her, một phim indie khác): văn bản mạch lạc, đúng chủ đề phim.

**Kết luận:** corpus dùng được, cần 2 bước lọc nhỏ trước khi build_review_index.py: (a) bỏ review < 5 từ, (b) dedupe review trùng trong cùng phim.

## 2. Độ phủ trên `V_I` của HyCoRec (hredial)

| | Số lượng | % trên 6515 item |
|---|---|---|
| Có trong catalog RevCore (bất kể có review hay không) | 5397 | 82.8% |
| **Có review thật** | **2104** | **32.3%** |

→ Đúng như đã báo trước đó: ~1/3 item nhận tín hiệu review trực tiếp; phần còn lại trông cậy vào scope G lan truyền qua entity chung (Eq. 4/5) — đúng động lực thiết kế của tài liệu.

## 3. Đối chiếu với dữ liệu HyCoRec tác giả đã build (`*_conv_idx_to_review_info.pkl`)

**Phát hiện quan trọng nhất của bước này:** trường `selected_entityIds` trong file review pkl của HyCoRec **không nằm trong không gian `entity2id.json` của chính hredial** (chỉ ~55% giá trị rơi vào khoảng `[0, n_entity)`, và khi rơi vào khoảng đó thì trỏ sai thực thể — vd. id `14102` bị hiểu nhầm thành diễn viên "Jackie Joseph"). Test thử với `word2index_redial2.json` của RevCore cũng cho câu vô nghĩa.

Dò ngược thì `key2index_3rd.json` bên trong `data/dataset/hredial/nltk/` **giống byte-for-byte** file cùng tên trong repo RevCore — bằng chứng trực tiếp là chung một dòng pipeline tiền xử lý. Từ đó thử `entity2entityId.pkl` của RevCore (không gian 64,362 thực thể, lớn hơn hẳn không gian 33,995-thực-thể riêng của hredial) — **giải mã đúng 100%** (24,574/24,574 lượt tham chiếu), ví dụ id `14102` → `The_Purge:_Anarchy` (một bộ phim thật, khớp ngữ cảnh).

Với cách giải mã đúng: 2882/24,574 lượt tham chiếu trỏ vào chính một bộ phim (không phải người/địa điểm); trong số đó **1039 phim (36.1%) đã có review thật trong `movieid2review_dict.pkl`** — sát với tỷ lệ nền tự nhiên của toàn catalog RevCore (39.2%), tức không có thiên lệch/trùng khớp giả. Ví dụ đối chiếu trực tiếp: `conv_idx=0` trong `test_conv_idx_to_review_info.pkl` gắn với thực thể `The_Purge:_Anarchy`, và review độc lập từ RevCore cho đúng phim này là: *"I almost didn't bother with this sequel. The first movie was close but no cigar down to rather lazy and ill thought plotting and since this movie was also written and directed by James DeMonaco..."* — đúng phim, đúng đạo diễn thật (James DeMonaco).

**Kết luận:** RevCore và dữ liệu review mà tác giả HyCoRec tự build **cùng nguồn/cùng không gian thực thể DBpedia** (không phải hai nguồn không liên quan tình cờ join được qua URI). Dùng RevCore cho Nhánh 2 là dùng đúng loại dữ liệu tác giả HyCoRec từng dựa vào, không phải nguồn thay thế tùy tiện.

## 4. Việc chưa giải quyết (không chặn Nhánh 2)

- Chưa xác định được đúng scheme token hoá của `selected_infoListListInt` (thử cả `token2id.json` lẫn `word2index_redial2.json` đều ra câu xáo trộn vô nghĩa — có thể đây vốn là tập từ-khoá-nổi-bật không giữ thứ tự, không phải câu nguyên). Không quan trọng vì Nhánh 2 tự xây `E_top⁺(R_i)` từ văn bản gốc RevCore (mạch lạc, đọc được), không tái dùng file này.
- TG-ReDial/Douban: vẫn chưa có nguồn (đã ghi ở kế hoạch), Nhánh 2 giới hạn ReDial.

## Khuyến nghị

**Đủ điều kiện để hoàn tất Nhánh 2 trên ReDial** với `movieid2review_dict.pkl` + `id2entity.pkl` làm nguồn `R_i`, sau khi lọc review rác (<5 từ) và dedupe.
