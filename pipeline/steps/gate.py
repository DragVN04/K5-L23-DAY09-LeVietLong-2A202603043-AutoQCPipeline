"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
from ..common import ERROR, WARN  # noqa: F401


def decide(issues, image_stems, cfg, reviewed=frozenset()):

    # 1. Đếm ERROR
    n_errors = sum(
        1 for issue in issues
        if issue.severity == ERROR
    )

    # 2. Lấy tập hợp các ảnh đang có
    valid_images = set(image_stems)
    reviewed_images = set(reviewed)

    # 3. Đếm ảnh có WARN chưa được duyệt
    warn_images = set()

    for issue in issues:
        if issue.severity != WARN:
            continue

        if issue.image not in valid_images:
            continue

        if issue.image in reviewed_images:
            continue

        warn_images.add(issue.image)

    n_warn_images = len(warn_images)
    n_images = len(valid_images)

    # 4. Tính tỉ lệ WARN
    if n_images > 0:
        warn_ratio = n_warn_images / n_images
    else:
        warn_ratio = 0.0

    warn_ratio = round(warn_ratio, 4)

    # 5. Quyết định Quality Gate
    threshold = cfg["gate"]["warn_ratio_hold"]

    if n_errors > 0:
        decision = "Rework"
        reasons = [
            f"Còn {n_errors} lỗi ERROR bắt buộc phải sửa."
        ]

    elif warn_ratio >= threshold:
        decision = "Hold"
        reasons = [
            f"Có {n_warn_images}/{n_images} ảnh còn cảnh báo "
            f"({warn_ratio:.2%}), đạt hoặc vượt ngưỡng "
            f"{threshold:.2%}."
        ]

    else:
        decision = "Release"
        reasons = [
            f"Không còn ERROR; có {n_warn_images}/{n_images} "
            f"ảnh cảnh báo chưa duyệt ({warn_ratio:.2%}), "
            f"dưới ngưỡng {threshold:.2%}."
        ]

    # 6. Trả kết quả cho pipeline
    return {
        "decision": decision,
        "n_errors": n_errors,
        "n_warn_images": n_warn_images,
        "n_images": n_images,
        "warn_ratio": warn_ratio,
        "reasons": reasons
    }
