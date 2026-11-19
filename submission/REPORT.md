# Báo cáo Day 09 — Lê Việt Long, 2A202603043

## 1. Kết luận trước

Pipeline AutoQC được triển khai để kiểm tra, tự sửa và đánh giá chất lượng dữ liệu gán nhãn YOLO, sử dụng 224 ảnh đầu vào.

Với cấu hình cuối cùng (`phash_dist = 16`, `second_opinion.conf = 0.60`), bước Intake tiếp nhận 193/224 ảnh, loại 31 ảnh (13.84%). Vòng QC đầu tiên phát hiện 36 lỗi ERROR và 149 cảnh báo WARN.

Sau AutoFix và mô phỏng người duyệt với ngân sách 80 ảnh, Quality Gate không còn ERROR nhưng vẫn có 34/193 ảnh còn WARN chưa xử lý, tương ứng 17.62%.

**Kết luận: Hold.** Tỷ lệ cảnh báo chưa xử lý vượt ngưỡng cố định 10%. Dataset được xuất dưới dạng bản nháp phục vụ thử nghiệm, chưa đủ điều kiện phát hành.

## 2. Lỗi và phân việc

Trong vòng QC đầu tiên sau Intake, hệ thống phát hiện 36 ERROR và 149 WARN. AutoFix thực hiện 35 thay đổi, sau đó còn 6 ERROR cần xử lý ở bước người duyệt.

| Nhóm lỗi                                         | Ai xử lý       | Nguyên nhân                                               |
| ------------------------------------------------ | -------------- | --------------------------------------------------------- |
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO               | Máy            | Dòng nhãn có cấu trúc hoặc giá trị không hợp lệ           |
| DUP_BOX                                          | Máy            | Bounding box cùng class bị trùng                          |
| BOX_OUT_OF_IMAGE                                 | Máy            | Cắt box về giới hạn ảnh                                   |
| BAD_CLASS_NAME                                   | Máy            | Đồng bộ class trong data.yaml                             |
| DUP*IMAGE*\*, LEAK_TEST, BAD_SIZE, CORRUPT_IMAGE | Máy tại Intake | Loại ảnh không đạt yêu cầu                                |
| MISSING_LABEL_FILE                               | Người          | Không thể tự tạo nhãn đúng từ file thiếu                  |
| POSSIBLE_MISSING                                 | Người          | Model phát hiện vật thể nhưng không tìm thấy nhãn phù hợp |
| POSSIBLE_WRONG_CLASS                             | Người          | Model và nhãn không thống nhất class                      |
| POSSIBLE_FALSE_POS                               | Người          | Bbox đã gán không khớp dự đoán của model                  |
| POSSIBLE_LOOSE_BOX                               | Người          | Bbox có khả năng bị lỏng hoặc lệch                        |
| BOX_TINY, BOX_HUGE, EMPTY_LABEL                  | Người          | Cần đối chiếu kích thước và quy tắc gán nhãn              |

Hàng đợi người duyệt ưu tiên ảnh theo điểm rủi ro. Các trường hợp thiếu file nhãn và nghi vấn ngữ nghĩa quan trọng được ưu tiên để giảm nguy cơ đưa nhãn sai vào dữ liệu train.

Với ngân sách 80 ảnh, người duyệt không xử lý hết mọi cảnh báo. Sau quy trình, còn 34 ảnh có WARN chưa xử lý, khiến Quality Gate giữ trạng thái Hold.

## 3. Calibration — Hiệu chỉnh hai ngưỡng

| Tham số               | Khởi đầu | Đã chọn | Precision / Recall trước → sau                                       |
| --------------------- | -------: | ------: | -------------------------------------------------------------------- |
| duplicates.phash_dist |        2 |      16 | 1.00/0.50 → 1.00/0.83 (DUP_IMAGE_NEAR)                               |
| second_opinion.conf   |     0.25 |    0.60 | 0.56/0.83 → 0.62/0.83 (WRONG_CLASS); 0.50/1.00 → 0.57/1.00 (MISSING) |

**Lý do lựa chọn:**

Ngưỡng `phash_dist = 16` đạt F1 = 0.91 và Recall = 0.83 đối với ảnh gần trùng, cao hơn ngưỡng ban đầu, trong khi Precision vẫn đạt 1.00 trên tập calibration.

Ngưỡng `second_opinion.conf = 0.60` giúp cải thiện Precision của WRONG_CLASS và MISSING mà vẫn giữ Recall của hai loại lỗi này. Tuy nhiên, Recall của LOOSE_BOX giảm từ 0.40 xuống 0.20, thể hiện sự đánh đổi khi tăng confidence.

**Hạn chế calibration:** Tập calibration có 39 ảnh, chưa chắc đại diện cho toàn bộ dữ liệu raw. Sau khi thay đổi ngưỡng, số ảnh bị loại tại Intake tăng từ 19 lên 31. Điều này cần được kiểm chứng trực quan để tránh loại nhầm ảnh hợp lệ. Precision đạt 1.00 trên calibration không bảo đảm không có False Positive trên dữ liệu thực tế.

## 4. Quality Gate và điểm mù của pipeline

Quality Gate áp dụng quy tắc:

- Rework nếu còn ít nhất một ERROR.
- Hold nếu không còn ERROR nhưng tỷ lệ ảnh WARN chưa duyệt lớn hơn hoặc bằng 10%.
- Release nếu không còn ERROR và tỷ lệ WARN dưới 10%.

Trong lần kiểm thử với ngân sách người duyệt bằng 0, pipeline tiếp nhận 205 ảnh theo cấu hình lúc kiểm thử, phát hiện 36 ERROR ở vòng QC đầu. Sau 35 thay đổi tự động, vẫn còn 6 ERROR.

Pipeline đưa ra quyết định Rework và không xuất dataset, chứng minh cửa chặn có khả năng ngăn dữ liệu còn lỗi bắt buộc được phát hành.

**Hai điểm mù:**

1. Second Opinion phụ thuộc vào dự đoán YOLO26l. Model có thể bỏ sót vật thể hoặc nhầm class, khiến nhãn đúng bị cảnh báo hoặc nhãn sai không được phát hiện.
2. Phát hiện ảnh gần trùng bằng pHash phụ thuộc ngưỡng và đặc điểm ảnh. Biến đổi mạnh có thể khiến ảnh trùng bị bỏ sót; ngưỡng quá rộng có thể loại nhầm ảnh khác nhau.

## 5. Train — So sánh Raw và Clean

Chưa thực hiện phần train trên Google Colab. Vì vậy, chưa có số liệu mAP50, mAP50-95, Precision và Recall của model train trên raw và clean.

Chưa thể kết luận dataset sau AutoQC giúp cải thiện hiệu năng mô hình.

## 6. Quyết định và ghi chú sử dụng AI

**Quyết định cuối cùng: Hold.**

Cấu hình cuối cùng cho kết quả 0 ERROR nhưng vẫn còn 34/193 ảnh chưa xử lý cảnh báo (17.62%). Để đủ điều kiện Release, cần tiếp tục kiểm tra các ảnh còn WARN, ưu tiên trường hợp thiếu nhãn và sai class. Với mẫu số 193 ảnh không đổi, số ảnh còn WARN phải giảm xuống tối đa 19 ảnh để dưới ngưỡng 10%.

**Sử dụng AI hỗ trợ:**

- Sử dụng ChatGPT để giải thích yêu cầu và hỗ trợ triển khai bốn module: `label_format.py`, `gate.py`, `autofix.py`, `second_opinion.py`.
- Đọc và đối chiếu logic với docstring, unit test, sau đó kiểm tra bằng `pytest` (79 passed) và chạy pipeline trên dữ liệu thực tế.
- Trong quá trình triển khai, lệnh pytest từng chạy nhầm vào môi trường ảo Day 08 dẫn đến lỗi thiếu module `pytest`. Sau khi kiểm tra đường dẫn Python, chuyển về môi trường Day 09 và chạy lại thành công. Đây là ví dụ về việc cần xác minh môi trường và kết quả thực thi thay vì chỉ dựa vào hướng dẫn AI.

Không sửa thủ công các báo cáo do pipeline sinh ra, không thay đổi quy tắc Gate cố định và không sử dụng kết quả train chưa thực hiện.
