"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    fixes = []

    # Chỉ xử lý lỗi ERROR, không tự sửa WARN
    errors = [i for i in issues if i.severity == ERROR]

    # Các lỗi yêu cầu xóa cả ảnh và nhãn
    remove_codes = {
        "LEAK_TEST",
        "CORRUPT_IMAGE",
        "BAD_SIZE"
    }

    # Các lỗi yêu cầu xóa dòng nhãn
    drop_codes = {
        "BAD_FORMAT",
        "BAD_CLASS_ID",
        "BOX_ZERO",
        "DUP_BOX"
    }

    # -----------------------------------------
    # 1. Loại ảnh bị lỗi nghiêm trọng
    # -----------------------------------------
    removed_images = set()

    for issue in errors:
        if not (
            issue.code in remove_codes
            or issue.code.startswith("DUP_IMAGE_")
        ):
            continue

        stem = issue.image

        # Mỗi ảnh chỉ xử lý một lần
        if stem in removed_images:
            continue

        image_path = ds.image_path(stem)

        if image_path is None:
            continue

        label_path = ds.label_path(stem)

        image_path.unlink()

        if label_path.exists():
            label_path.unlink()

        removed_images.add(stem)

        fixes.append({
            "image": stem,
            "line": "",
            "action": "remove_image",
            "before": issue.code,
            "after": "Đã loại ảnh và nhãn"
        })

    # -----------------------------------------
    # 2. Chuyển ảnh thiếu nhãn cho người
    # -----------------------------------------
    missing_images = set()

    for issue in errors:
        if issue.code != "MISSING_LABEL_FILE":
            continue

        stem = issue.image

        if stem in removed_images or stem in missing_images:
            continue

        if ds.image_path(stem) is None:
            continue

        if ds.label_path(stem).exists():
            continue

        missing_images.add(stem)

        fixes.append({
            "image": stem,
            "line": "",
            "action": "needs_label",
            "before": "(không có file)",
            "after": "chuyển người gán"
        })

    # -----------------------------------------
    # 3. Sửa tên class trong data.yaml
    # -----------------------------------------
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        yaml_path = ds.yaml_path

        if yaml_path.exists():
            data = yaml.safe_load(
                yaml_path.read_text(encoding="utf-8")
            ) or {}

            old_names = data.get("names")
            new_names = list(cfg["classes"])

            if old_names != new_names:
                data["names"] = new_names

                yaml_path.write_text(
                    yaml.safe_dump(
                        data,
                        allow_unicode=True,
                        sort_keys=False
                    ),
                    encoding="utf-8"
                )

                fixes.append({
                    "image": "data.yaml",
                    "line": "",
                    "action": "rename_classes",
                    "before": str(old_names),
                    "after": str(new_names)
                })

    # -----------------------------------------
    # 4. Gom lỗi theo ảnh và số dòng gốc
    # -----------------------------------------
    line_issues = defaultdict(lambda: defaultdict(set))

    for issue in errors:
        if issue.image in removed_images:
            continue

        if issue.line is None:
            continue

        if issue.code in drop_codes or issue.code == "BOX_OUT_OF_IMAGE":
            line_issues[issue.image][issue.line].add(issue.code)

    # -----------------------------------------
    # 5. Xử lý từng file nhãn
    # -----------------------------------------
    for stem, issue_by_line in sorted(line_issues.items()):
        label_path = ds.label_path(stem)

        if not label_path.exists():
            continue

        original_lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        updated_lines = []
        changed = False

        for line_no, text in enumerate(original_lines, start=1):
            codes = issue_by_line.get(line_no, set())

            # 5.1. Không có lỗi -> giữ nguyên
            if not codes:
                updated_lines.append(text)
                continue

            # 5.2. Lỗi phải xóa dòng
            if codes & drop_codes:
                fixes.append({
                    "image": stem,
                    "line": line_no,
                    "action": "drop_line",
                    "before": text,
                    "after": ""
                })

                changed = True
                continue

            # 5.3. Box tràn ảnh -> clip
            if "BOX_OUT_OF_IMAGE" in codes:
                try:
                    class_id, cx, cy, w, h = parse_yolo_line(text)
                except ValueError:
                    updated_lines.append(text)
                    continue

                # Chuyển từ cx,cy,w,h sang 4 mép
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                # Cắt các mép về [0,1]
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))

                new_w = x2 - x1
                new_h = y2 - y1

                # Box nằm hoàn toàn ngoài ảnh
                if new_w <= 0 or new_h <= 0:
                    fixes.append({
                        "image": stem,
                        "line": line_no,
                        "action": "drop_line",
                        "before": text,
                        "after": ""
                    })
                    changed = True
                    continue

                # Chuyển lại định dạng YOLO
                new_cx = (x1 + x2) / 2
                new_cy = (y1 + y2) / 2

                new_text = (
                    f"{class_id} "
                    f"{new_cx:.8f} {new_cy:.8f} "
                    f"{new_w:.8f} {new_h:.8f}"
                )

                updated_lines.append(new_text)

                if new_text != text:
                    fixes.append({
                        "image": stem,
                        "line": line_no,
                        "action": "clip_box",
                        "before": text,
                        "after": new_text
                    })
                    changed = True

                continue

            # Các dòng còn lại giữ nguyên
            updated_lines.append(text)

        # 6. Chỉ ghi file khi có thay đổi
        if changed:
            label_path.write_text(
                "\n".join(updated_lines) + "\n",
                encoding="utf-8"
            )

    return fixes

