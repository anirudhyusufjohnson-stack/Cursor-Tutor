"""Icon matching by nearest-neighbour search. This is a SENSOR: it turns pixels into a symbol such as Icon_paperclip.

Every icon you teach the app is stored as a few example images. To classify an icon under the cursor,
each example is turned into numbers (a 64x64 grey image and a colour histogram) and compared with the
new icon. The best-matching label wins if its similarity is high enough and clearly beats the runner-up."""
import re
import time
from pathlib import Path

import cv2
import numpy as np

SIZE = 64


def _prep(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    small = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    # hue + saturation only: ignoring brightness keeps the colour signature stable when an icon lights up on hover
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).astype(np.float32)
    return gray, hist


def augment(pil_image):
    """The example itself plus slightly brighter, darker and blurrier copies (data augmentation),
    so the classifier still works when an icon lights up under the mouse or is drawn a little blurry."""
    from PIL import ImageEnhance, ImageFilter

    img = pil_image.convert("RGB")
    yield img
    yield ImageEnhance.Brightness(img).enhance(1.18)
    yield ImageEnhance.Brightness(img).enhance(0.82)
    yield img.filter(ImageFilter.GaussianBlur(1.0))


def similarity(a, b):
    """Similarity 0-1 between two prepared icons: shape (pixel correlation) + colour (histogram)."""
    ncc = cv2.matchTemplate(a[0], b[0], cv2.TM_CCOEFF_NORMED)[0, 0]
    if not np.isfinite(ncc):
        ncc = 1.0 if np.array_equal(a[0], b[0]) else 0.0
    col = cv2.compareHist(a[1], b[1], cv2.HISTCMP_CORREL)
    return 0.6 * max(float(ncc), 0.0) + 0.4 * max(float(col), 0.0)


def safe_label(label):
    return re.sub(r"[^a-z0-9_-]+", "-", label.lower()).strip("-") or "item"


class IconStore:
    def __init__(self, folder, threshold=0.78, margin=0.04):
        self.folder = Path(folder)
        self.threshold = threshold
        self.margin = margin
        self.samples = {}  # label -> list of prepared icons
        self.reload()

    def reload(self):
        from PIL import Image

        self.samples = {}
        if not self.folder.exists():
            return
        for d in sorted(p for p in self.folder.iterdir() if p.is_dir()):
            for f in sorted(d.glob("*.png")):
                try:
                    for variant in augment(Image.open(f)):
                        self.samples.setdefault(d.name, []).append(_prep(variant))
                except Exception:
                    continue

    def count(self):
        """Number of icons taught (each is stored with a few augmented copies)."""
        return sum(len(list(d.glob("*.png"))) for d in self.folder.iterdir() if d.is_dir()) if self.folder.exists() else 0

    def add(self, label, pil_image):
        """Save an example of an icon under a label and learn it immediately."""
        label = safe_label(label)
        d = self.folder / label
        d.mkdir(parents=True, exist_ok=True)
        pil_image.convert("RGB").save(d / f"{int(time.time() * 1000)}.png")
        for variant in augment(pil_image):
            self.samples.setdefault(label, []).append(_prep(variant))
        return label

    def scores(self, pil_image):
        """Best similarity per taught label, highest first: [(score, label), ...]."""
        if not self.samples:
            return []
        probe = _prep(pil_image)
        return sorted(
            ((max(similarity(probe, s) for s in samples), label) for label, samples in self.samples.items()),
            reverse=True,
        )

    @staticmethod
    def decide(per_label, threshold, margin):
        """Pick a label from sorted scores, or None when nothing is similar enough or the top two are too close."""
        if not per_label:
            return None
        best_score, best_label = per_label[0]
        if best_score < threshold:
            return None
        if len(per_label) > 1 and best_score - per_label[1][0] < margin:
            return None  # too close to call: better to say nothing than to guess
        return best_label, float(best_score)

    def classify(self, pil_image):
        """Return (label, score) for the best match, or None when nothing is similar enough."""
        return self.decide(self.scores(pil_image), self.threshold, self.margin)
