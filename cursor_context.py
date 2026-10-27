"""Capture what is under the mouse cursor: screenshots + window / UI-element info."""
import base64
import io
import platform
from dataclasses import dataclass, field

import mss
import pyautogui
from PIL import Image, ImageDraw

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    import psutil

    _user32 = ctypes.windll.user32
    _user32.WindowFromPoint.argtypes = [wintypes.POINT]
    _user32.WindowFromPoint.restype = wintypes.HWND
    _user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    _user32.GetAncestor.restype = wintypes.HWND
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]


@dataclass
class CursorContext:
    x: int
    y: int
    window_title: str = ""
    process_name: str = ""
    ui_element: str = ""
    images_b64: list = field(default_factory=list)  # [overview, zoomed crop], JPEG base64 (used for thumbnails)
    region: object = None            # PIL image: raw (no red ring) area around the cursor, used by the algorithms
    region_cursor: tuple = (0, 0)    # cursor position inside `region`, in pixels

    def describe(self) -> str:
        bits = []
        if self.process_name:
            bits.append(f"app/process: {self.process_name}")
        if self.window_title:
            bits.append(f"window title: {self.window_title!r}")
        if self.ui_element:
            bits.append(f"UI element under cursor: {self.ui_element}")
        return "; ".join(bits) or "no extra window info available"


def window_under_point(x, y):
    """Return (window_title, process_name) of the top-level window under (x, y).
    Only implemented on Windows; elsewhere the AI relies on the screenshot alone."""
    if not IS_WINDOWS:
        return "", ""
    try:
        hwnd = _user32.WindowFromPoint(wintypes.POINT(int(x), int(y)))
        root = _user32.GetAncestor(hwnd, 2)  # GA_ROOT
        n = _user32.GetWindowTextLengthW(root)
        buf = ctypes.create_unicode_buffer(n + 1)
        _user32.GetWindowTextW(root, buf, n + 1)
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(root, ctypes.byref(pid))
        return buf.value, psutil.Process(pid.value).name()
    except Exception:
        return "", ""


def _ui_element_under(x, y):
    """Windows UI Automation: name/type of the exact button or control under the cursor."""
    if not IS_WINDOWS:
        return ""
    try:
        from pywinauto import Desktop

        info = Desktop(backend="uia").from_point(int(x), int(y)).element_info
        name = (info.name or "").strip()
        ctype = info.control_type or "control"
        return f"{ctype}: {name}" if name else ctype
    except Exception:
        return ""


def _encode(img, max_side):
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def capture_context(pos=None, use_uia=True):
    """Screenshot the monitor under `pos` (default: current cursor) and describe the target."""
    x, y = pos if pos else pyautogui.position()
    x, y = int(x), int(y)

    with mss.mss() as sct:
        mon = next(
            (
                m
                for m in sct.monitors[1:]
                if m["left"] <= x < m["left"] + m["width"] and m["top"] <= y < m["top"] + m["height"]
            ),
            sct.monitors[1],
        )
        shot = sct.grab(mon)
        full = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    # Convert cursor position to image pixels (handles HiDPI / Retina scaling).
    sx = full.width / mon["width"]
    sy = full.height / mon["height"]
    cx = int((x - mon["left"]) * sx)
    cy = int((y - mon["top"]) * sy)

    # 1) Overview of the whole screen with a red ring around the cursor.
    overview = full.copy()
    d = ImageDraw.Draw(overview)
    r = max(24, full.width // 80)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=max(4, r // 5))
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 0, 0))

    # 2) Zoomed crop around the cursor so small icons stay readable.
    cw, ch = int(700 * sx), int(450 * sy)
    left = min(max(cx - cw // 2, 0), max(full.width - cw, 0))
    top = min(max(cy - ch // 2, 0), max(full.height - ch, 0))
    crop = full.crop((left, top, min(left + cw, full.width), min(top + ch, full.height)))
    raw_region = crop.copy()  # keep a clean copy: the algorithms must not see the red ring
    cd = ImageDraw.Draw(crop)
    lx, ly = cx - left, cy - top
    cd.ellipse([lx - 14, ly - 14, lx + 14, ly + 14], outline=(255, 0, 0), width=3)

    title, proc = window_under_point(x, y)
    return CursorContext(
        x=x,
        y=y,
        window_title=title,
        process_name=proc,
        ui_element=_ui_element_under(x, y) if use_uia else "",
        images_b64=[_encode(overview, 1024), _encode(crop, 768)],
        region=raw_region,
        region_cursor=(lx, ly),
    )
