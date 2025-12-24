import cv2
import numpy as np
import glog as log

def compute_center_crop_box(src_w: int, src_h: int, ar_w: int, ar_h: int):
    """Compute a centered crop box to match the target aspect ratio.

    Returns (x0, y0, x1, y1) in pixel coordinates.
    """
    if src_w <= 0 or src_h <= 0 or ar_w <= 0 or ar_h <= 0:
        return 0, 0, src_w, src_h
    src_ar = float(src_w) / float(src_h)
    tgt_ar = float(ar_w) / float(ar_h)
    if abs(src_ar - tgt_ar) < 1e-6:
        return 0, 0, src_w, src_h
    if src_ar > tgt_ar:
        # too wide -> crop left/right
        new_w = int(round(src_h * tgt_ar))
        x0 = max(0, (src_w - new_w) // 2)
        return x0, 0, x0 + new_w, src_h
    else:
        # too tall -> crop top/bottom
        new_h = int(round(src_w / tgt_ar))
        y0 = max(0, (src_h - new_h) // 2)
        return 0, y0, src_w, y0 + new_h


def center_crop(img: np.ndarray, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def center_crop_and_resize_bgr(img_bgr: np.ndarray, target_wh=(640, 480), aspect_ratio=(4, 3)) -> np.ndarray:
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = compute_center_crop_box(w, h, aspect_ratio[0], aspect_ratio[1])
    if (x0, y0, x1, y1) != (0, 0, w, h):
        # one-time info for unusual sizes may be helpful
        log.debug(f"center crop from {w}x{h} to {(x1-x0)}x{(y1-y0)} for AR {aspect_ratio}")
    cropped = center_crop(img_bgr, (x0, y0, x1, y1))
    tw, th = int(target_wh[0]), int(target_wh[1])
    if tw > 0 and th > 0:
        resized = cv2.resize(cropped, (tw, th))
    else:
        resized = cropped
    return resized


def center_crop_and_resize_rgb(img_bgr: np.ndarray, target_wh=(640, 480), aspect_ratio=(4, 3)) -> np.ndarray:
    bgr = center_crop_and_resize_bgr(img_bgr, target_wh=target_wh, aspect_ratio=aspect_ratio)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

