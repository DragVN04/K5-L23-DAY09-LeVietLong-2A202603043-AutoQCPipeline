"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401

def run(ds, cfg):
    issues = []

    for image_path in ds.image_files():
        # Lấy tên file nhưng bỏ phần mở rộng. VD: img_0001.jpg → img_0001
        stem = image_path.stem
        lines = ds.read_label_lines(stem)

        if lines is None:
            continue

        # Khởi tạo danh sách box của từng ảnh
        previous_boxes = []

        for line_no, text in lines:

            # 1. Kiểm tra định dạng
            try:
                class_id, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BAD_FORMAT",
                    image=stem,
                    line=line_no,
                    message="Dòng nhãn YOLO sai định dạng"
                ))
                continue

            # 2. Kiểm tra class ID
            if not (0 <= class_id < len(ds.classes)):
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BAD_CLASS_ID",
                    image=stem,
                    line=line_no,
                    message=f"Class ID {class_id} không hợp lệ"
                ))
                continue

            # 3. Kiểm tra kích thước box
            if w <= 0 or h <= 0:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BOX_ZERO",
                    image=stem,
                    line=line_no,
                    message="Bounding box có kích thước không hợp lệ"
                ))
                continue

            # 4. Kiểm tra box tràn ảnh
            x_min = cx - w / 2
            x_max = cx + w / 2
            y_min = cy - h / 2
            y_max = cy + h / 2

            overflow = max(
                0.0,
                -x_min,
                x_max - 1.0,
                -y_min,
                y_max - 1.0
            )

            tolerance = cfg["label_format"]["overflow_tol"]

            if overflow > tolerance:
                issues.append(Issue(
                    check="label_format",
                    severity=ERROR,
                    code="BOX_OUT_OF_IMAGE",
                    image=stem,
                    line=line_no,
                    message=f"Bounding box tràn ảnh: {overflow:.4f}",
                    extra={"overflow": overflow}
                ))
                
             # 5. Kiểm tra box trùng
            current_box = (cx, cy, w, h)

            for old_class, old_box in previous_boxes:
                if old_class != class_id:
                    continue

                iou = iou_xywh(current_box, old_box)

                if iou > cfg["label_format"]["dup_box_iou"]:
                    issues.append(Issue(
                        check="label_format",
                        severity=ERROR,
                        code="DUP_BOX",
                        image=stem,
                        line=line_no,
                        message=f"Box trùng với box trước đó, IoU={iou:.3f}"
                    ))
                    break

            previous_boxes.append((class_id, current_box))

             # 6. Kiểm tra box quá nhỏ
            size = ds.image_size(stem)

            if size is not None:
                img_w, img_h = size

                short_side = min(
                    w * img_w,
                    h * img_h
                )

                if short_side < cfg["label_format"]["min_box_px"]:
                    issues.append(Issue(
                        check="label_format",
                        severity=WARN,
                        code="BOX_TINY",
                        image=stem,
                        line=line_no,
                        message=f"Box quá nhỏ: {short_side:.2f}px"
                    ))

            # 7. Kiểm tra box quá lớn
            if w * h > cfg["label_format"]["max_box_frac"]:
                issues.append(Issue(
                    check="label_format",
                    severity=WARN,
                    code="BOX_HUGE",
                    image=stem,
                    line=line_no,
                    message="Bounding box chiếm diện tích quá lớn"
                ))

    return issues
