"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    issues = []

    # 1. Lấy dự đoán từ model nếu chưa được truyền vào
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    params = cfg["second_opinion"]

    # 2. Duyệt từng ảnh
    for image_path in ds.image_files():
        stem = image_path.stem

        if stem not in preds:
            continue

        lines = ds.read_label_lines(stem)

        if lines is None:
            continue

        labels = ds.valid_boxes(stem)
        predictions = preds[stem]

        size = ds.image_size(stem)

        if size is None:
            continue

        img_w, img_h = size

        # 3. Lọc dự đoán theo confidence và diện tích
        filtered_preds = []

        for pred in predictions:
            cls, conf, cx, cy, w, h = pred

            if conf < params["conf"]:
                continue

            area_px = w * img_w * h * img_h

            if area_px < params["min_area_px"]:
                continue

            filtered_preds.append(pred)

        # 4. So sánh từng dự đoán với nhãn
        for pred in filtered_preds:
            pred_cls, conf, cx, cy, w, h = pred
            pred_box = (cx, cy, w, h)

            candidates = []

            for label in labels:
                label_cls = label[0]
                label_box = label[1:]

                iou = iou_xywh(pred_box, label_box)

                if iou >= params["iou_missing"]:
                    candidates.append((label_cls, iou))

            # 4.1. Không có nhãn khớp -> có thể thiếu nhãn
            if not candidates:
                issues.append(Issue(
                    check="second_opinion",
                    severity=WARN,
                    code="POSSIBLE_MISSING",
                    image=stem,
                    message="Model phát hiện vật thể chưa có nhãn",
                    extra={"conf": conf}
                ))
                continue

            # 4.2. Tìm nhãn cùng class
            same_class_ious = [
                iou for cls, iou in candidates
                if cls == pred_cls
            ]

            if same_class_ious:
                best_iou = max(same_class_ious)

                if best_iou < params["iou_loose"]:
                    issues.append(Issue(
                        check="second_opinion",
                        severity=WARN,
                        code="POSSIBLE_LOOSE_BOX",
                        image=stem,
                        message="Bounding box có thể bị lỏng hoặc lệch",
                        extra={
                            "conf": conf,
                            "iou": best_iou
                        }
                    ))

                # Đã có nhãn cùng class, không xét wrong class
                continue

            # 4.3. Không có nhãn cùng class
            other_class_ious = [
                iou for cls, iou in candidates
                if cls != pred_cls
            ]

            if other_class_ious:
                best_iou = max(other_class_ious)

                if best_iou >= params["iou_match"]:
                    issues.append(Issue(
                        check="second_opinion",
                        severity=WARN,
                        code="POSSIBLE_WRONG_CLASS",
                        image=stem,
                        message="Model và nhãn không thống nhất class",
                        extra={
                            "conf": conf,
                            "iou": best_iou
                        }
                    ))

        # 5. Kiểm tra nhãn có thể là False Positive
        # Dùng fp_conf, không lọc min_area_px
        fp_predictions = [
            pred for pred in predictions
            if pred[1] >= params["fp_conf"]
        ]

        # Duyệt dòng gốc để bảo toàn line number
        for line_no, text in lines:
            try:
                label = parse_yolo_line(text)
            except ValueError:
                continue

            label_cls = label[0]

            if not (0 <= label_cls < len(ds.classes)):
                continue

            label_box = label[1:]

            matched = False

            for pred in fp_predictions:
                pred_box = pred[2:]

                iou = iou_xywh(label_box, pred_box)

                if iou >= params["iou_missing"]:
                    matched = True
                    break

            if not matched:
                issues.append(Issue(
                    check="second_opinion",
                    severity=WARN,
                    code="POSSIBLE_FALSE_POS",
                    image=stem,
                    line=line_no,
                    message="Nhãn không khớp dự đoán nào của model"
                ))

    return issues
