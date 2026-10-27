"""Text recognition (OCR): reads the words around the mouse cursor.

Uses RapidOCR, a small neural network (text detector + text recogniser) that runs offline on the CPU.
If it is not installed, falls back to Tesseract when available, otherwise the app still works from
window titles and learned icons."""
from dataclasses import dataclass

import numpy as np


@dataclass
class TextBox:
    text: str
    box: tuple      # (x0, y0, x1, y1) in region pixels
    conf: float     # 0-1


def distance_to_box(point, box):
    """Distance from a point to a rectangle (0 when the point is inside it)."""
    x, y = point
    x0, y0, x1, y1 = box
    dx = max(x0 - x, 0, x - x1)
    dy = max(y0 - y, 0, y - y1)
    return (dx * dx + dy * dy) ** 0.5


def nearest_texts(boxes, cursor, limit=5, max_distance=160):
    """Text boxes closest to the cursor, nearest first: [(TextBox, distance_px), ...]."""
    scored = [(b, distance_to_box(cursor, b.box)) for b in boxes]
    scored = [s for s in scored if s[1] <= max_distance]
    scored.sort(key=lambda s: (s[1], -s[0].conf))
    return scored[:limit]


class OCR:
    def __init__(self, enabled=True):
        self.engine = None
        self.name = "off"
        self.kind = None
        if not enabled:
            return
        try:
            from rapidocr_onnxruntime import RapidOCR

            self.engine = RapidOCR()
            self.kind = "rapid"
            self.name = "RapidOCR (neural network, offline)"
            return
        except Exception:
            pass
        try:
            # Newer package name, needed on Python 3.13+ (pip install rapidocr onnxruntime)
            import logging

            from rapidocr import RapidOCR

            self.engine = RapidOCR()
            logging.getLogger("RapidOCR").setLevel(logging.ERROR)  # hide its info/warning chatter
            self.kind = "rapid3"
            self.name = "RapidOCR (neural network, offline)"
            return
        except Exception:
            pass
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            self.engine = pytesseract
            self.kind = "tesseract"
            self.name = "Tesseract (offline)"
        except Exception:
            self.name = "not installed"

    @property
    def available(self):
        return self.engine is not None

    def read(self, pil_image):
        """Return a list of TextBox for the words found in a PIL image."""
        if not self.available:
            return []
        try:
            if self.kind == "rapid":
                result, _ = self.engine(np.array(pil_image.convert("RGB")))
                out = []
                for box, text, score in result or []:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    out.append(TextBox(str(text).strip(), (min(xs), min(ys), max(xs), max(ys)), float(score)))
                return [t for t in out if t.text]
            if self.kind == "rapid3":
                result = self.engine(np.array(pil_image.convert("RGB")))
                if result is None or result.boxes is None:
                    return []
                out = []
                for box, text, score in zip(result.boxes, result.txts, result.scores):
                    xs = [float(p[0]) for p in box]
                    ys = [float(p[1]) for p in box]
                    out.append(TextBox(str(text).strip(), (min(xs), min(ys), max(xs), max(ys)), float(score)))
                return [t for t in out if t.text]
            data =self.engine.image_to_data(pil_image, output_type=self.engine.Output.DICT)
            out = []
            for i, txt in enumerate(data["text"]):
                try:
                    conf = float(data["conf"][i]) / 100.0
                except ValueError:
                    continue
                if txt.strip() and conf > 0.3:
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    out.append(TextBox(txt.strip(), (x, y, x + w, y + h), conf))
            return out
        except Exception:
            return []
