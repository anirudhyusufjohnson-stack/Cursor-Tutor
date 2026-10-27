"""Background thread that notices when the mouse stops moving ("hovering")."""
import math
import threading
import time

import pyautogui


class HoverWatcher(threading.Thread):
    def __init__(self, dwell_seconds, in_overlay, on_move, on_dwell, enabled):
        super().__init__(daemon=True)
        self.dwell = dwell_seconds
        self.in_overlay = in_overlay  # pos -> bool: is the cursor over our own window?
        self.on_move = on_move        # pos -> None: remember last position outside our window
        self.on_dwell = on_dwell      # pos -> None: the cursor has been still long enough
        self.enabled = enabled        # () -> bool: is auto-explain switched on?
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        anchor = tuple(pyautogui.position())
        since = time.time()
        fired = False
        while not self._stop_event.is_set():
            time.sleep(0.1)
            pos = tuple(pyautogui.position())
            if self.in_overlay(pos):
                continue  # ignore the tutor window itself
            self.on_move(pos)
            if math.dist(pos, anchor) > 10:  # moved: restart the timer
                anchor, since, fired = pos, time.time(), False
            elif not fired and self.enabled() and time.time() - since >= self.dwell:
                fired = True
                self.on_dwell(pos)
