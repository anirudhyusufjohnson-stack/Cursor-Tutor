"""Find the exact icon or button under the cursor with classic computer vision.

Edge detection (Canny) + morphological dilation + contour analysis: the strokes of one icon are
merged into a single blob, and the smallest blob that contains the cursor is taken as the target."""
import cv2
import numpy as np

MIN_SIZE = 20
MAX_SIZE = 260
FALLBACK_SIZE = 96


def segment_target(region, cursor):
    """Return (box, method): box = (x0, y0, x1, y1) in region pixels around the cursor.
    method is 'contour' when an object was isolated, or 'fixed' for a plain square fallback."""
    w, h = region.size
    cx, cy = int(cursor[0]), int(cursor[1])
    gray = cv2.cvtColor(np.array(region.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if not (MIN_SIZE <= bw <= MAX_SIZE and MIN_SIZE <= bh <= MAX_SIZE):
            continue
        if not (x <= cx <= x + bw and y <= cy <= y + bh):
            continue
        area = bw * bh
        if best is None or area < best[0]:
            best = (area, (x, y, x + bw, y + bh))
    if best:
        return best[1], "contour"

    half = FALLBACK_SIZE // 2
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, w), min(cy + half, h)
    return (x0, y0, x1, y1), "fixed"


def crop_box(region, box):
    return region.crop(tuple(int(v) for v in box))
