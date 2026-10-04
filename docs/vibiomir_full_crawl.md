# Crawl toàn bộ ViBioMIR

Dataset riêng tư đang xây dựng:
https://www.kaggle.com/datasets/duymaingoc/vibiomir-rag-corpus

Notebook điều phối chạy trên Kaggle:
https://www.kaggle.com/code/duymaingoc/vibiomir-cloud-coordinator

## Phạm vi và nhãn

Theo quy định BTC người dùng cung cấp, đội thi tự thu thập toàn bộ nguồn trong
`links_corpus.parquet`. Nhãn dương là trường **`id` integer của tài liệu** trong
bảng này. Passage `rag:*` chỉ phục vụ truy hồi nội bộ. Không đánh số lại document,
không dùng row index làm ID và không tạo official chunk IDs.

Tool lập kế hoạch cho đúng **4.394.718 URL**, chia **4.395 shard** tối đa 1.000
dòng/shard. Mỗi shard giữ một khoảng row cố định trong catalogue đã pin SHA256;
ID gốc vẫn lấy từ cột `id`, kể cả khi không liên tục. Bộ điều phối chạy hai
notebook CPU riêng tư, không phân phối đồng thời các shard chung hostname.

Mỗi notebook crawl tối đa 6 giờ rồi dừng cấp URL mới, hoàn tất request đang chạy,
lưu SQLite và export Parquet. Bộ điều phối tự chạy tiếp từ checkpoint của phiên
trước. Shard đầu lưu checkpoint sớm sau 60 giây để kiểm tra quy trình resume;
các phiên tiếp theo dùng budget 6 giờ. Catalogue gốc được mount từ dataset version
1 trên Kaggle, tránh tải lại 35 MB từ Hugging Face cho hàng nghìn phiên.

Checkpoint đầu tiên đã xuất 25 tài liệu và 116 passage. Phiên chạy tiếp dùng
output version 1 của `duymaingoc/vibiomir-rag-full-01-a` làm input; đây vẫn là
kết quả từng phần, không phải dataset hoàn chỉnh.
Log live đã xác nhận `resume: True` và tiếp tục lấy ID 608, 609, 610… sau 25
nguồn đã lưu, không chạy lại từ đầu.

Kaggle quy định tối đa 12 giờ mỗi phiên CPU và 20 GB output lưu trong
`/kaggle/working`. State của từng shard và Parquet được giữ riêng để phù hợp giới
hạn này. [Thông số Kaggle](https://www.kaggle.com/docs/notebooks).

## Các sửa lỗi thu thập

- Giữ protocol của URL gốc. Một số nguồn HTTP không hỗ trợ HTTPS hoặc có certificate
  sai; ép HTTPS làm mất các tài liệu vẫn truy cập được bình thường.
- Retry lỗi mạng, HTTP 429/5xx với backoff và `Retry-After`. Không lặp lại 404/403
  một cách vô hạn.
- Parser robots hỗ trợ group hợp nhất, wildcard, ưu tiên path và crawl delay.
  HTTP lỗi khi tải robots được ghi riêng, không giả thành một `Disallow` rule.
  Cache lỗi robots có TTL ngắn, tránh mất cả domain vì một timeout.
- Chromium thực thi JavaScript thông thường cho trang cần trình duyệt, gồm cả
  endpoint robots trả cookie/reload page. Giữ browser context theo origin để dùng
  lại session. Không đưa CAPTCHA/challenge page vào corpus.
- Sửa HTML đóng `</html>` trước `<head>`, decode Unicode trước parsing, fallback
  JSON-LD `articleBody` và selector có phạm vi cụ thể. Không lấy toàn bộ menu/footer
  làm nội dung thay thế khi extractor thất bại.
- Giữ bảng trạng thái của **mọi URL đã xử lý**, kể cả lỗi và tài liệu thiếu.

Đã khôi phục thực tế các URL ở Báo Tây Ninh, food.39.net, gan.39.net, Lao Động,
health.people.com.cn, Wujue và Long Châu. Đây là xác nhận trên những URL đã kiểm
tra, không phải cam kết mọi URL trong các domain đều lấy được.

Robots tuân theo [RFC 9309](https://www.rfc-editor.org/rfc/rfc9309.html) qua
[Protego](https://github.com/scrapy/protego). Trang bị xoá hoặc nguồn thực sự từ
chối truy cập vẫn cần phục hồi từ URL thay thế/snapshot phù hợp hoặc dữ liệu nguồn.

## Điều phối CPU quanh ngày

`ViBioMIR continuous CPU crawl` dùng GitHub Actions chạy một nhịp điều phối mỗi
5 phút, ở mọi giờ trong ngày. Worker crawl vẫn chạy **trên Kaggle CPU**;
metadata đặt rõ `enable_gpu=false`, `enable_tpu=false`. Khi worker hoàn tất một
shard hoặc phiên 6 giờ, nhịp điều phối kế tiếp nhận/checksum kết quả rồi gửi
shard tiếp theo hoặc phiên resume. Không còn budget điều phối 9 giờ/ngày, không
cần bật máy local hay bấm chạy lại notebook.

Workflow: `.github/workflows/vibiomir-continuous.yml`.
Entrypoint: `scripts/vibiomir_cloud_tick.py`. Credential OAuth Kaggle nằm trong
**encrypted GitHub Actions secret** `KAGGLE_OAUTH_CREDENTIALS`; chỉ được ghi vào
credential file mode 0600 của runner tạm thời. Không có token trong git, notebook,
dataset, cache hoặc file trạng thái. SDK tự tạo access token mới từ OAuth refresh
credential để việc crawl dài ngày không phụ thuộc một access token 12 giờ.

Workflow không chạy hai coordinator cùng lúc (`concurrency`, không huỷ job đang
lưu tiến độ). Nó chờ **phiên native v10 đang chạy** kết thúc, rồi thay notebook
Daily bằng notebook trạng thái không có Internet/Secret hoặc thao tác với dataset.
Sau đó nhận checkpoint cloud mới nhất; worker cũ vẫn được theo dõi và nhận kết quả.
Các lần Daily còn được Kaggle kích hoạt chỉ in trạng thái bàn giao, không crawl hay
ghi đè tiến độ. Đây là bàn giao tự động, không cần người dùng tắt lịch thủ công.

Mỗi lần gửi worker lưu submission intent trước API push; nếu runner bị ngắt sau
khi Kaggle nhận push, nhịp sau tìm đúng submission key và nhận lại version đã chạy.
Worker lỗi được thử lại sau cooldown, giữ giới hạn hostname và slot; shard đã lưu
không bị cộng đúp. Master index cập nhật tối đa mỗi 30 phút trong lúc xây dựng,
còn kết quả đã thu được lưu vào partition/checkpoint ngay. Chỉ trạng thái tổng hợp
không chứa credential được ghi vào GitHub job summary. Bản tổng hợp trong repo
được cập nhật mỗi 28 ngày để lịch public không bị vô hiệu vì repo không hoạt động.

"Quanh ngày" ở đây nghĩa là mọi khung giờ đều được điều phối và worker được nối
phiên, không có khoảng nghỉ hằng ngày do code đặt. Không phải SLA không gián đoạn:
queue Kaggle, quota, thời gian khởi tạo, lỗi nguồn và GitHub schedule chậm có thể
tạo khoảng chờ ngắn. Giữ nhịp tiếp theo để tự phục hồi; không dùng vòng lặp local.
[GitHub scheduled workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule),
[GitHub encrypted secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

## Bộ điều phối Daily trước khi bàn giao

`ViBioMIR Cloud Coordinator` nhận kết quả worker, kiểm tra checksum/counts, cập
nhật dataset và lưu tiến độ. **Kaggle native scheduler** khởi động phiên tiếp
theo hằng ngày. Phiên bình thường có budget 9 giờ, chừa khoảng cách với giới hạn 12 giờ.
Trạng thái lưu trong dataset riêng tư `duymaingoc/vibiomir-crawl-control`;
không phụ thuộc terminal hoặc máy local. Bộ điều phối local đã được dừng.

**Thiết lập một lần:** mở notebook điều phối → **Add-ons → Secrets** → thêm
hoặc bật secret **`KAGGLE_API_TOKEN`**. Trong **Settings → Schedule a notebook
to run**, chọn **Daily**, rồi **Save & Run All**. Token cần quyền
ghi dataset và thực thi notebook. Đăng nhập CLI local không tự cấp quyền cho
notebook trên Kaggle. Không đặt token vào cell, dataset hay gửi vào chat.
[Kaggle Secrets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md#using-secrets-in-kernels).

Kaggle API `kernels push` không mang quyền Secrets sang phiên mới; đã xác minh
thực tế: Saved Run tạo từ UI đọc được Secret, phiên kế tiếp tạo từ API trả HTTP
400 khi đọc Secret. Vì vậy không dùng chuỗi tự push notebook điều phối.
Native scheduler giữ quyền Secrets và hỗ trợ chạy hằng ngày.
[Kaggle Scheduling Notebooks](https://www.kaggle.com/docs/notebooks),
[Kaggle Secrets và API](https://www.kaggle.com/product-feedback/666571).

Nếu chưa đọc được secret, notebook dừng với `authentication_required`. Mỗi phiên
tải **checkpoint cloud mới nhất** qua API, không dùng cố định snapshot cũ trong
Inputs. Phiên kiểm tra đầu có budget 60 giây rồi lưu `waiting_for_schedule`; sau
đó phiên hằng ngày dùng budget 9 giờ. Worker đã gửi vẫn chạy khi controller nghỉ;
kết quả được nhận ở phiên điều phối tiếp theo. Lịch này có thời gian nghỉ giữa
các ngày; không đồng nghĩa crawl liên tục 24 giờ. Nếu Kaggle huỷ phiên hoặc token
hết quyền, cần xem Logs và chạy lại sau khi sửa lỗi. Không cần máy local online.

Chỉ chạy một notebook điều phối. Giữ nguyên chế độ **Private**. Không cần mở
trình duyệt sau khi một Saved Run đã khởi động và đọc được Secrets.
Code notebook: `notebooks/ViBioMIR_Cloud_Coordinator.ipynb`.

**Xác minh thực tế ngày 2026-10-03:** Saved Run v9 đọc được Secret, khôi phục
checkpoint v6, nhận thêm kết quả và kết thúc thành công. Checkpoint control v10
đã được tải lại và kiểm tra SHA-256/size: `waiting_for_schedule`, generation 2;
6.000 URL đã xử lý, 5.973 tài liệu và 20.678 chunks. Hai worker vẫn `RUNNING`
sau khi controller kết thúc. Người dùng đã bật Daily và metadata có nhãn
`scheduled`; lần chạy hằng ngày tiếp theo chưa được quan sát tại thời điểm này.
Đây là xác nhận chạy/khôi phục/lưu tiến độ trên Kaggle, chưa phải hoàn tất corpus.
Biên bản local: `outputs/vibiomir_cloud_v9_verification/verification.json`.

**Xác minh tiếp ngày 2026-10-04:** phiên điều phối v10 đã tự chạy, đọc Secret,
khôi phục checkpoint v10 và dùng budget 32.400 giây (9 giờ). Metadata của
controller và hai worker đang chạy đều `enable_gpu=false`, `enable_tpu=false`,
`machine_shape=None`: toàn bộ crawl dùng CPU. Checkpoint v59 đã kiểm tra SHA-256
và size; chứa 18.000 URL, 17.962 tài liệu, 59.117 chunks; 18 shard hoàn tất,
hai shard đang chạy, không có shard lỗi. Plan vẫn bao phủ toàn bộ 4.394.718 URL
trong 4.395 shard, không phải crawl mẫu. Lịch Daily tiếp tục qua các phiên đến
khi plan kết thúc; khoảng nghỉ giữa các phiên vẫn tồn tại.
Biên bản: `outputs/vibiomir_cloud_oct04_verification/verification.json`.

Lệnh sau chỉ dùng một lần để chuyển tiến độ từ bộ điều phối local đã dừng:

```bash
python scripts/deploy_vibiomir_cloud.py --owner YOUR_USERNAME
```

Chạy lớn cần thời gian dài: riêng hostname lớn nhất có 963.438 URL. Với khoảng
cách tối thiểu 1 giây/request, riêng nguồn này cần ít nhất khoảng **11 ngày**,
chưa tính thời gian phản hồi, robots delay, retry và thời gian khởi tạo notebook.
Đây là cận dưới từ cấu hình hiện tại, không phải ETA. Hai worker không bỏ giới
hạn theo hostname để tăng tốc một nguồn.

## File dataset và kiểm tra hoàn chỉnh

| File | Nội dung |
| --- | --- |
| `query.parquet`, `links_corpus.parquet` | Hai bảng gốc, giữ schema integer của BTC |
| `documents/documents-NNNNN.parquet` trong partition | Văn bản, URL, title, `official_id` integer |
| `chunks/chunks-NNNNN.parquet` trong partition | Passage, `official_doc_id` integer và `chunk_id` nội bộ |
| `sources/sources-NNNNN.parquet` trong partition | ID/URL/status/error của từng nguồn đã xử lý |
| `manifests/shard-NNNNN.json` | Counts, range, source/config/code hash và checksum file |
| `dataset_manifest.json` | Coverage toàn collection, số nguồn thiếu và trạng thái |

Metadata object được serialize vào cột `metadata_json`. Mỗi file được kiểm tra
SHA256 và số dòng Parquet trước khi nhập vào collection. Bộ điều phối xuất các
file cập nhật lên Kaggle sau kết quả đầu tiên, tối đa mỗi 24 giờ sau đó và khi
kết thúc; việc publish luôn giữ dataset riêng tư và Parquet nguyên bản.

Corpus được lưu thành các dataset `vibiomir-rag-part-NNNN` nhỏ; master
`vibiomir-rag-corpus` chứa catalogue, coverage và manifest ánh xạ file sang
**`dataset_ref` đã pin version**. Mỗi partition tối đa 4 shard theo cấu hình,
thường dưới 1 GiB. Một shard lớn được giữ nguyên, không cắt nội dung cho vừa
partition. Bộ điều phối chỉ cache partition đang cập nhật, tránh tích luỹ cả
corpus trên đĩa notebook. Snapshot không chứa token, output worker hay cache corpus.

Kaggle giới hạn 50 file ở cấp gốc và 200 GB mỗi dataset. Vì vậy tool gom các
shard vào bốn thư mục trên, upload bằng `dir_mode="zip"`; Kaggle giải nén giữ
cấu trúc thư mục. Manifest đã chuẩn hóa đường dẫn theo gốc dataset.
[Giới hạn dataset Kaggle](https://www.kaggle.com/docs/datasets).

`acquisition_complete` nghĩa mọi URL đã có trạng thái. **`content_complete` chỉ
đúng khi mọi URL đều có nội dung.** `state: needs_source_recovery` nghĩa đã xử lý
hết catalogue nhưng vẫn thiếu nội dung; không coi đó là corpus đủ 100%. Shard
đang chạy tiếp có `pending_urls > 0`. Nguồn lỗi tạm thời được thử lại một vòng
riêng sau lần crawl đầu; 404/robots disallow không bị retry vô hạn.

Đọc dataset không cần ghép toàn bộ JSONL hay nạp tất cả documents vào RAM:

```python
import pyarrow.dataset as ds
from pathlib import Path

root = Path("/kaggle/input/YOUR_PARTITION_PATH")
chunks = ds.dataset([str(p) for p in root.glob("chunks/chunks-*.parquet")], format="parquet")
for batch in chunks.to_batches(batch_size=1000):
    rows = batch.to_pylist()
    # Embed/index rows in batches; retain official_doc_id for document ranking.
```

Để đọc toàn collection, dùng danh sách file và `dataset_ref` trong manifest
master. Chỉ đọc file được manifest chỉ định, tránh tính trùng output từng phần
còn giữ ở version/partition cũ. Không glob `chunks` trực tiếp trong master.
Helper `src.data.kaggle_cloud_runner.iter_cloud_chunks(manifest_path, api, cache_dir)`
tải từng partition, kiểm tra checksum và trả PyArrow batches; cache được dọn
sau mỗi partition. Đây là cách đọc toàn corpus mà không cần giữ tất cả trong RAM.

`src.data.full_crawl.document_rankings` gộp passage theo điểm cao nhất của từng
tài liệu và trả integer document IDs. Format đóng gói bài thi phải theo schema
submission BTC; validator legacy doc/chunk/string chưa phải format ViBioMIR mới.
