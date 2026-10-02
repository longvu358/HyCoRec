# Corpus review cho TG-ReDial (Nhánh 2) — kết quả tìm kiếm

Đối chiếu bằng tải thật + đo độ phủ trên `htgredial` (V_I = 31 985 item).

## Ứng viên

| Corpus | Quy mô | Dùng được? |
|---|---|---|
| **`dirtycomputer/douban_movie_review`** (HF; cùng nội dung `douban-movie-1000w`) | 10 269 phim, 10 310 989 comment, 2.3 GB CSV / 6 parquet shard; cột `Movie_Name, Star(1–5), Comment, Like, Date, Score…` | **Có — chọn cái này** |
| Douban Movie Short Comments (Kaggle `utmhikari`) | 2M comment nhưng chỉ **28 phim** | Không (độ phủ ≈ 0) |
| Zenodo 8415635 | 224 phim, CC-BY 4.0 | Không (quá nhỏ) |
| `dengfuping/douban-movies-dataset` | ~2.5k phim, chỉ metadata (không có comment) | Không |
| MovieData-10M (`moviedata.csuldw.com`) | 140k phim, 4.4M comment | Không tải được (DNS lỗi lúc thử); chưa đánh giá |

## Độ phủ trên TG-ReDial (đo thật trên 10 269 tên phim)

Khớp theo tên: bỏ `<a>`, bỏ chú thích cuối `（…）`, phía Douban lấy phần tiếng Trung trước khoảng trắng đầu tiên (tên Douban dạng `记忆碎片 Memento`).

| | Giá trị |
|---|---|
| item distinct được nhắc trong hội thoại | 11 968 |
| **có review** | **5 572 (46.6 %)** |
| **tính theo lượt nhắc** | **59.0 %** |
| top-100 / top-1000 / top-3000 item phổ biến | 83 % / 74 % / 63 % |
| tên khớp mơ hồ (>1 phim cùng tên) | 22 |

So với ReDial/RevCore (32.3 % V_I, chỉ ~1/3 item): **TG-ReDial với corpus này phủ tốt hơn**, đặc biệt ở item hay xuất hiện. Cách khớp mới là heuristic đơn giản, còn thiếu (ví dụ `切腹（1962年…）`, `芝加哥`); khớp thêm bằng năm/đạo diễn sẽ tăng thêm.

Ngoài ra V_I của TG (31 985) có cả nhạc/sách… (hub `音乐作品`, `单曲`), corpus này chỉ có phim → các item đó vẫn nhờ scope G lan truyền.

## Khác biệt so với RevCore cần xử lý trong `build_review_index.py`

1. **Comment ngắn**: median **22 ký tự** (p10 = 5, p90 = 92) so với 154 từ của RevCore. Mỗi phim trung vị ~1 075 comment, nhiều hơn RevCore (median 1), bù được độ ngắn.
2. **Có sẵn `Star` 1–5** → dùng làm nhãn sentiment thay cho VADER (không dùng được cho tiếng Trung): `Star ≥ 4` ⇒ positive. Không cần mô hình sentiment, và tránh lỗi VADER.
3. **Entity linking tiếng Trung**: thay `build_surface_index` (Title Case, regex Latin) bằng khớp chuỗi trên tên entity pkuseg (bỏ chú thích `（…）`), longest-match theo ký tự; loại item V_I và tên quá ngắn (≥2 ký tự để khỏi nhiễu).
4. Tách câu bằng dấu câu Trung (`。！？；`) hoặc coi mỗi comment là một đơn vị (do ngắn).
5. Lọc: bỏ comment <5 ký tự, dedupe trong cùng phim (`Like` có thể dùng làm trọng số).

## Bản quyền / giấy phép (đã kiểm tra 2026-10-02)

- **Repo HF `dirtycomputer/douban_movie_review`**: `license = null`, không có dataset card, chỉ 1 file CSV. Người upload không tuyên bố giấy phép nào → mặc định **bảo lưu mọi quyền**, không có quyền tái phân phối được cấp.
- **Nội dung gốc là comment người dùng Douban**, crawl 2019-10-05 (không phải dữ liệu do uploader tạo). [Điều khoản sử dụng Douban](https://www.douban.com/about/agreement?page=2024-01-05) (bản 2024-01-05, đọc trực tiếp):
  - cl. 9.4: không sao chép/bán/dùng *"cho bất kỳ mục đích thương mại nào"*;
  - cl. 10.4: không sao chép, sửa đổi, **không "chế tác tác phẩm phái sinh"**;
  - cl. 10.2: người dùng giữ quyền tác giả, chỉ cấp cho Douban giấy phép không độc quyền;
  - **không có ngoại lệ nghiên cứu/phi thương mại**. Trang bản quyền Douban hướng tới liên hệ `bd-team@douban.com` để xin phép (theo kết quả tìm kiếm, tôi chưa mở lại trang đó).
- Dataset có cùng nguồn gốc với DMSC/TG-ReDial (TG-ReDial cũng dựng từ dữ liệu Douban), nên cộng đồng CRS vẫn dùng, nhưng điều đó **không biến nó thành được cấp phép**.

**Đánh giá thực tế (không phải tư vấn pháp lý):**
- Dùng nội bộ để dựng index và chạy thí nghiệm: rủi ro thấp/phổ biến trong nghiên cứu.
- `review_index.json` **chỉ chứa id item, id entity và tf** (không có văn bản comment) → có thể chia sẻ kèm code mà không phát tán comment gốc; đây là cách làm an toàn nhất để tái lập.
- **Không** commit/phát tán raw parquet (đã nằm trong `.gitignore` qua `**/data/reviews`).
- Khi viết bài: ghi rõ nguồn (Douban comments qua HF mirror), nói rõ chỉ dùng cho nghiên cứu, và nếu cần phát hành thì xin phép Douban hoặc để người đọc tự tải.

## Kết quả build (`python build_review_index.py -d htgredial --k_rev 10`)

Raw đặt ở `data/reviews/raw/douban1000w/` (+ `PROVENANCE.md`); output `data/reviews/htgredial/review_index.json` (+ `review_index_stats.json`).

| | |
|---|---|
| item V_I khớp phim Douban | 8 361 / 31 985 (26 %); 50 tên mơ hồ bị bỏ |
| comment dùng / positive (Star ≥ 4) | 7 883 354 / 3 234 629 |
| positive có ≥1 entity | 2 083 579 |
| item có `E_top` | 8 358 (mean 10 entity) |
| entity phân biệt trong index | 9 540 |

**Lần build đầu bị nhiễu** và đã sửa: KG kiểu Baike có entity cho từ thường (`喜欢（胡杨林歌曲）`, `he`, `on`, `to`) nên 56 % slot rơi vào entity xuất hiện ở >5 % item (kết quả vô nghĩa kiểu `记忆碎片 → [he, to, on, 喜欢…]`). Sửa bằng hai bước:
1. Loại surface form không có chữ Hán và entity có chú thích loại bài hát/album/sách/thương hiệu/công ty/game…
2. Xếp hạng theo **tf × idf** (giá trị lưu vẫn là tf thô). Chỉ áp cho `htgredial` (mặc định), `hredial` giữ nguyên tf để index cũ không đổi (`--idf/--no-idf`).

Sau sửa: slot trên entity phổ biến chỉ còn 2.9 %. Ví dụ: `记忆碎片 → 倒叙, 失忆, 彩色, 盗梦空间, 禁闭岛…`; `降临 → 外星人, 科幻片, 硬科幻, 哲学…`; `鬼子来了 → 抗日, 抗战, 日军, 汉奸, 讽刺…`.

**Còn nhiễu nhỏ**: vài entity trùng từ thường vẫn lọt (`妻子（…电视剧）`, `孩子（同名微电影）`) vì phần chú thích không thuộc danh sách loại bị chặn; có thể bổ sung `_ZH_NON_ASPECT_PAREN` nếu cần.

## Rủi ro / chưa xác minh

- Dataset có thể trùng lặp với nguồn mà TG-ReDial dùng (reviews cũng lấy từ Douban) — chưa kiểm chứng, nhưng đó là cùng nền tảng nên phù hợp.
- Độ phủ 46.6 % tính trên chuẩn hoá tên đơn giản; một phần "trượt" là do khác tên, không phải do thiếu phim.

## Bước tiếp theo

1. (Server) `python build_collective.py -d htgredial --use_review` để dựng `data/collective/htgredial_unified_full`, rồi đổi `tgredial.yaml`: `use_review_hypergraph: true`, `review_index_path: data/reviews/htgredial/review_index.json`, `collective_path` trỏ tới bản `_full`. **Chưa làm** — chưa chạy build_collective/train.
2. Chạy ablation A8/A9 trên TG để xem review có giúp không (item phủ 26 % V_I, 59 % lượt nhắc → kỳ vọng tác động nằm ở item phổ biến).
