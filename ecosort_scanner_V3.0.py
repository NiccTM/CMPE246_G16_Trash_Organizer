import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

"""
EcoSort G16 - RPi 5 + Freenove 8MP + LCD

UBC campus bins: Garbage, Recycling, Returnables, Compost, E-Waste

Controls:
  B     = capture background (do this first with nothing in frame)
  SPACE = manual scan
  S     = stats screen
  F     = fullscreen toggle
  Q     = quit

Auto-scan triggers when you hold an object still for ~1.5s

Camera setup:
  - ribbon cable into CAM0, blue stripe away from board
  - in /boot/firmware/config.txt add:
      camera_auto_detect=0
      dtoverlay=imx219,cam0

LCD wiring (PCF8574 backpack):
  GND -> Pin 6, VCC -> Pin 2, SDA -> Pin 3, SCL -> Pin 5

LED wiring (each with 220 ohm resistor to GND):
  Garbage (red)      -> GPIO 17
  Recycling (blue)   -> GPIO 27
  Returnables (green) -> GPIO 22
  Compost (orange)   -> GPIO 23
  E-Waste (yellow)   -> GPIO 24

Needs a .env file with ROBOFLOW_API_KEY=your_key
"""

import cv2
import numpy as np
import os
import time
import threading
from collections import defaultdict
from dotenv import load_dotenv
from roboflow import Roboflow
from picamera2 import Picamera2

# LCD init - falls back gracefully if not plugged in
LCD_AVAILABLE = False
lcd = None
LCD_I2C_ADDR = 0x27  # change to 0x3f if thats what i2cdetect shows
LCD_COLS = 16
LCD_ROWS = 2

try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD(i2c_expander='PCF8574', address=LCD_I2C_ADDR,
                  port=1, cols=LCD_COLS, rows=LCD_ROWS)
    lcd.clear()
    LCD_AVAILABLE = True
    print(f"LCD connected at 0x{LCD_I2C_ADDR:02x}")
except Exception as e:
    print(f"LCD not available ({e}) — running without LCD")

# LED setup - one per bin, uses gpiozero (works on Pi 5)
LED_AVAILABLE = False
leds = {}

try:
    from gpiozero import LED as GPIOLED
    leds = {
        "GARBAGE":     GPIOLED(17),  # red
        "RECYCLING":   GPIOLED(27),  # blue
        "RETURNABLES": GPIOLED(22),  # green
        "COMPOST":     GPIOLED(23),  # orange
        "E-WASTE":     GPIOLED(24),  # yellow
    }
    LED_AVAILABLE = True
    print("LEDs initialized on GPIO 17, 27, 22, 23, 24")
except Exception as e:
    print(f"LEDs not available ({e}) — running without LEDs")

# Config
CAMERA_PORT = 0
CAPTURE_W = 1280
CAPTURE_H = 720
CONFIDENCE = 15
SNAP_PATH = "/tmp/ecosort_snap.jpg"
HEADLESS = False
DEBUG = True

RESULT_DISPLAY = 8000    # how long to show result (ms)
NODET_DISPLAY = 4000
LCD_SCROLL_DELAY = 0.35

# background subtraction params
BG_DIFF_THRESH = 35
MIN_OBJECT_AREA = 3000
OBJECT_PAD = 60
BG_ADAPT_RATE = 0.02

# auto-scan - fires after object stays still this many frames (~1.5s at 30fps)
AUTO_SCAN_STABLE_FRAMES = 45
AUTO_SCAN_MOVE_THRESH = 30   # px - how much bbox can move and still count as "still"
AUTO_SCAN_COOLDOWN = 5.0     # seconds between auto-scans

CONFIDENCE_FLOOR = 40  # below this % we show "unsure"

LIGHT_TOO_DARK = 40
LIGHT_TOO_BRIGHT = 240

AUTO_BG_RECAPTURE_AFTER = 5  # re-capture bg after this many misses in a row

# classes the model returns that are actual specific objects (get bonus weight)
SPECIFIC_CLASSES = {
    "can", "aluminum", "bottle", "glass", "plastic container",
    "cap or lid", "paper", "cardboard boxes", "cardboard boxes and cartons",
    "battery", "styrofoam", "food & others",
}

# these come from background clutter (cables, desk, etc)
BACKGROUND_PRONE = {
    "plastic bag or wrapper", "plastic bag", "utensils and straw",
    "utensils/straw", "trash",
}

SAME_BIN_TIE_THRESHOLD = 0.15

# higher = more specific label, used for tie-breaking display name
LABEL_SPECIFICITY = {
    "can": 10, "aluminum": 10, "bottle": 10, "glass": 10,
    "battery": 10, "styrofoam": 10,
    "plastic container": 8,
    "paper": 7, "cardboard boxes": 7, "cardboard boxes and cartons": 7,
    "food & others": 7,
    "cap or lid": 3, "container": 3,
    "trash": 1, "plastic bag or wrapper": 1, "utensils and straw": 1,
}

BIN_COLORS_BGR = {
    "GARBAGE":     (128, 128, 128),
    "RECYCLING":   (255, 180, 0),
    "RETURNABLES": (0, 180, 0),
    "COMPOST":     (0, 165, 255),
    "E-WASTE":     (180, 105, 255),
}

BIN_DISPLAY = {
    "GARBAGE":     {"icon": "X", "label": "Garbage"},
    "RECYCLING":   {"icon": "R", "label": "Recycling"},
    "RETURNABLES": {"icon": "$", "label": "Returnables"},
    "COMPOST":     {"icon": "C", "label": "Compost"},
    "E-WASTE":     {"icon": "E", "label": "E-Waste"},
}


# --- LCD scrolling stuff ---
# scrolls long text on the 16-char LCD in a background thread

_scroll_thread = None
_scroll_stop = threading.Event()

def lcd_show(line1, line2=""):
    if not LCD_AVAILABLE:
        return
    _stop_scroll()
    if len(line1) > LCD_COLS or len(line2) > LCD_COLS:
        _start_scroll(line1, line2)
    else:
        _lcd_write_static(line1, line2)

def _lcd_write_static(line1, line2=""):
    try:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1[:LCD_COLS].ljust(LCD_COLS))
        if line2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2[:LCD_COLS].ljust(LCD_COLS))
    except Exception as e:
        print(f"  LCD write error: {e}")

def _scroll_worker(line1, line2):
    pad = " " * 3
    text1 = (line1 + pad + line1) if len(line1) > LCD_COLS else line1.ljust(LCD_COLS)
    text2 = (line2 + pad + line2) if len(line2) > LCD_COLS else line2.ljust(LCD_COLS)
    scroll1 = len(line1) > LCD_COLS
    scroll2 = len(line2) > LCD_COLS
    _lcd_write_static(line1, line2)
    # wait a bit before scrolling starts
    for _ in range(6):
        if _scroll_stop.is_set():
            return
        time.sleep(LCD_SCROLL_DELAY)
    offset1, offset2 = 0, 0
    wrap1 = len(line1) + len(pad) if scroll1 else 1
    wrap2 = len(line2) + len(pad) if scroll2 else 1
    while not _scroll_stop.is_set():
        try:
            lcd.cursor_pos = (0, 0)
            if scroll1:
                lcd.write_string("".join(
                    text1[(offset1 + i) % len(text1)] for i in range(LCD_COLS)))
            else:
                lcd.write_string(text1[:LCD_COLS])
            lcd.cursor_pos = (1, 0)
            if scroll2:
                lcd.write_string("".join(
                    text2[(offset2 + i) % len(text2)] for i in range(LCD_COLS)))
            else:
                lcd.write_string(text2[:LCD_COLS])
        except Exception:
            pass
        if scroll1:
            offset1 = (offset1 + 1) % wrap1
        if scroll2:
            offset2 = (offset2 + 1) % wrap2
        time.sleep(LCD_SCROLL_DELAY)

def _start_scroll(line1, line2):
    global _scroll_thread
    _scroll_stop.clear()
    _scroll_thread = threading.Thread(
        target=_scroll_worker, args=(line1, line2), daemon=True)
    _scroll_thread.start()

def _stop_scroll():
    global _scroll_thread
    if _scroll_thread and _scroll_thread.is_alive():
        _scroll_stop.set()
        _scroll_thread.join(timeout=1.0)
    _scroll_thread = None


def check_lighting(frame):
    """Returns a warning string if the frame is too dark/bright, None if ok"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < LIGHT_TOO_DARK:
        return f"Too dark ({mean_brightness:.0f})"
    if mean_brightness > LIGHT_TOO_BRIGHT:
        return f"Too bright ({mean_brightness:.0f})"
    return None


def led_show(bin_name):
    """Turn on the LED for the given bin, turn off all others"""
    if not LED_AVAILABLE:
        return
    for name, led in leds.items():
        if name == bin_name:
            led.on()
        else:
            led.off()

def led_off():
    """Turn off all LEDs"""
    if not LED_AVAILABLE:
        return
    for led in leds.values():
        led.off()

def led_blink(bin_name, times=3, delay=0.2):
    """Blink the LED for a bin a few times then leave it on"""
    if not LED_AVAILABLE:
        return
    led = leds.get(bin_name)
    if not led:
        return
    led_off()
    for _ in range(times):
        led.on()
        time.sleep(delay)
        led.off()
        time.sleep(delay)
    led.on()

def led_cleanup():
    """Turn off all LEDs on exit"""
    if not LED_AVAILABLE:
        return
    led_off()


class EcoSortScanner:

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            sys.exit("ERROR: Add ROBOFLOW_API_KEY=... to a .env file next to this script.")

        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("taco-ihjgk").project("yolov8-trash-detections-kgnug")
            self.model = project.version(11).model
            print("Roboflow model loaded")
        except Exception as e:
            sys.exit(f"Roboflow connection failed: {e}")

        # maps roboflow class names to our UBC bins
        self.bin_logic = {
            "GARBAGE": [
                "styrofoam", "plastic bag", "plastic bag or wrapper",
                "utensils/straw", "utensils and straw", "straw",
                "utensils", "trash", "cigarette",
            ],
            "RECYCLING": [
                "paper", "cardboard boxes", "cardboard boxes and cartons",
                "cardboard", "cap or lid", "plastic container", "container",
            ],
            "RETURNABLES": [
                "can", "aluminum", "aluminium", "tin", "metal",
                "bottle", "glass", "beverage",
            ],
            "COMPOST": [
                "food & others", "food",
            ],
            "E-WASTE": [
                "battery",
            ],
        }

        self.camera = None
        self.bg_frame = None       # float32 grayscale background
        self.bg_frame_u8 = None    # uint8 version for diffing
        self.bg_ready = False

        self.stats = defaultdict(int)
        self.total_scans = 0
        self.total_unsure = 0

        # for the mini score display in top right
        self._last_result_bin = None
        self._last_result_label = None
        self._last_result_score = 0.0
        self._last_result_time = 0.0

        # auto-scan tracking
        self._prev_bbox = None
        self._stable_count = 0
        self._last_scan_time = 0.0

        self._consecutive_no_det = 0
        self._fullscreen = False
        self._window_name = "EcoSort G16"

    def _get_bin(self, label):
        label_lower = label.lower()
        for bin_name, keywords in self.bin_logic.items():
            if any(k in label_lower for k in keywords):
                return bin_name
        return "GARBAGE"

    # camera stuff

    def _open_camera(self):
        cam = Picamera2(camera_num=CAMERA_PORT)
        config = cam.create_preview_configuration(
            main={"format": "XRGB8888", "size": (CAPTURE_W, CAPTURE_H)},
            controls={"FrameDurationLimits": (33333, 100000)}
        )
        cam.configure(config)
        cam.start()
        time.sleep(1.5)
        self.camera = cam
        print(f"IMX219 camera opened on CAM{CAMERA_PORT} ({CAPTURE_W}x{CAPTURE_H})")

    def _read_frame(self):
        raw = self.camera.capture_array()
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def _close_camera(self):
        if self.camera:
            self.camera.stop()
            print("Camera released.")

    # background subtraction

    def _capture_background(self):
        """Average 15 frames to get a clean background reference"""
        print("Capturing background (keep scene empty)...")
        lcd_show("Capturing BG...", "Keep area clear")

        frames = []
        for _ in range(15):
            frames.append(self._read_frame())
            time.sleep(0.08)

        avg = np.mean(frames, axis=0).astype(np.uint8)
        gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        self.bg_frame = blurred.astype(np.float32)
        self.bg_frame_u8 = blurred
        self.bg_ready = True
        self._consecutive_no_det = 0

        print("Background captured.")
        lcd_show("BG captured!", "Place object")
        time.sleep(1.0)

    def _adapt_background(self, frame, has_object):
        """Slowly blend current frame into bg when theres no object in the way"""
        if self.bg_frame is None or has_object:
            return
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0
        ).astype(np.float32)
        cv2.accumulateWeighted(gray, self.bg_frame, BG_ADAPT_RATE)
        self.bg_frame_u8 = self.bg_frame.astype(np.uint8)

    def _find_object_bbox(self, frame):
        """Diff current frame against background and find the biggest new thing"""
        if self.bg_frame_u8 is None:
            return None
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        diff = cv2.absdiff(self.bg_frame_u8, gray)
        _, thresh = cv2.threshold(diff, BG_DIFF_THRESH, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_OBJECT_AREA:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        fh, fw = frame.shape[:2]
        x1 = max(0, x - OBJECT_PAD)
        y1 = max(0, y - OBJECT_PAD)
        x2 = min(fw, x + w + OBJECT_PAD)
        y2 = min(fh, y + h + OBJECT_PAD)
        return (x1, y1, x2 - x1, y2 - y1)

    # auto-scan - checks if bbox has been stable long enough

    def _check_auto_scan(self, bbox):
        if not self.bg_ready:
            return False
        now = time.time()
        if now - self._last_scan_time < AUTO_SCAN_COOLDOWN:
            self._stable_count = 0
            return False
        if bbox is None:
            self._stable_count = 0
            self._prev_bbox = None
            return False
        if self._prev_bbox is None:
            self._prev_bbox = bbox
            self._stable_count = 1
            return False

        # check how far the bbox center moved
        cx1 = self._prev_bbox[0] + self._prev_bbox[2] // 2
        cy1 = self._prev_bbox[1] + self._prev_bbox[3] // 2
        cx2 = bbox[0] + bbox[2] // 2
        cy2 = bbox[1] + bbox[3] // 2
        dist = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

        if dist < AUTO_SCAN_MOVE_THRESH:
            self._stable_count += 1
        else:
            self._stable_count = 1
        self._prev_bbox = bbox
        return self._stable_count >= AUTO_SCAN_STABLE_FRAMES

    def _reset_auto_scan(self):
        self._stable_count = 0
        self._prev_bbox = None
        self._last_scan_time = time.time()

    def _enhance(self, frame):
        """CLAHE contrast boost on the L channel"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _bbox_is_centered(pred, img_w, img_h, tolerance=0.40):
        cx = pred["x"] / img_w
        cy = pred["y"] / img_h
        return abs(cx - 0.5) < tolerance and abs(cy - 0.5) < tolerance

    def _pick_best_label(self, scores):
        """If top two classes go to the same bin and are close in score,
        pick the more descriptive one for display (e.g. 'can' over 'cap or lid')"""
        ranked = sorted(scores, key=scores.get, reverse=True)
        winner = ranked[0]
        winner_bin = self._get_bin(winner)
        if len(ranked) >= 2:
            runner_up = ranked[1]
            if (self._get_bin(runner_up) == winner_bin
                    and scores[winner] > 0
                    and (scores[winner] - scores[runner_up]) / scores[winner]
                    <= SAME_BIN_TIE_THRESHOLD):
                w_spec = LABEL_SPECIFICITY.get(winner.lower(), 5)
                r_spec = LABEL_SPECIFICITY.get(runner_up.lower(), 5)
                if r_spec > w_spec:
                    if DEBUG:
                        print(f"  Tie-break: '{runner_up}' over '{winner}'")
                    return winner_bin, runner_up
        return winner_bin, winner

    @staticmethod
    def _merge_cooccurrences(scores, counts):
        """The model sees the pull-tab on cans as 'cap or lid'. When both
        'can' and 'cap or lid' show up in the same scan, merge lid into can
        since its obviously the top of the can, not a separate object.
        Same idea for bottle + lid."""
        lid_keys = [k for k in scores if k.lower() == "cap or lid"]
        can_keys = [k for k in scores if k.lower() == "can"]
        bottle_keys = [k for k in scores if k.lower() == "bottle"]

        if lid_keys:
            lid_k = lid_keys[0]
            merge_target = None
            if can_keys:
                merge_target = can_keys[0]
            elif bottle_keys:
                merge_target = bottle_keys[0]

            if merge_target:
                scores[merge_target] += scores[lid_k]
                counts[merge_target] += counts[lid_k]
                del scores[lid_k]
                del counts[lid_k]
                print(f"  Merged 'cap or lid' into '{merge_target}'")

        # the model sometimes returns both "cardboard boxes and cartons" and
        # "cardboard boxes" for the same thing
        cb_long = [k for k in scores if k.lower() == "cardboard boxes and cartons"]
        cb_short = [k for k in scores if k.lower() == "cardboard boxes"]
        if cb_long and cb_short:
            scores[cb_short[0]] += scores[cb_long[0]]
            counts[cb_short[0]] += counts[cb_long[0]]
            del scores[cb_long[0]]
            del counts[cb_long[0]]

        return scores, counts

    @staticmethod
    def _score_to_pct(raw_score):
        """Map raw weighted score to 0-100%. Uses exponential curve so
        ~1.0 -> 40%, ~3.0 -> 75%, ~6.0 -> 92%"""
        pct = int(100 * (1 - np.exp(-raw_score / 3.0)))
        return max(0, min(100, pct))

    # main inference pipeline

    def _infer(self, frame):
        bbox = self._find_object_bbox(frame)
        if bbox is None:
            print("  BG subtraction found nothing, using center crop")
            h, w = frame.shape[:2]
            m = 0.20
            mx, my = int(w * m), int(h * m)
            bbox = (mx, my, w - 2 * mx, h - 2 * my)

        bx, by, bw, bh = bbox
        object_crop = frame[by:by+bh, bx:bx+bw]

        # also try a slightly wider crop for more context
        fh, fw = frame.shape[:2]
        ex = OBJECT_PAD
        wx1, wy1 = max(0, bx - ex), max(0, by - ex)
        wx2, wy2 = min(fw, bx + bw + ex), min(fh, by + bh + ex)
        wider_crop = frame[wy1:wy2, wx1:wx2]

        variants = [
            ("isolated", object_crop),
            ("enhanced", self._enhance(object_crop)),
            ("wider",    wider_crop),
        ]

        all_hits = []

        for var_name, img in variants:
            img_h, img_w = img.shape[:2]
            if img_h < 10 or img_w < 10:
                continue
            cv2.imwrite(SNAP_PATH, img)
            try:
                results = self.model.predict(SNAP_PATH, confidence=CONFIDENCE).json()
                preds = results.get("predictions", [])
            except Exception as e:
                print(f"    [{var_name}] error: {e}")
                continue

            if DEBUG:
                brief = [(p["class"], round(p["confidence"], 2)) for p in preds]
                print(f"    [{var_name:10s}] {brief}")

            for p in preds:
                cls = p["class"]
                conf = p["confidence"]
                w = 1.0
                if cls.lower() in SPECIFIC_CLASSES:
                    w *= 1.5
                if cls.lower() in BACKGROUND_PRONE:
                    w *= 0.4
                if self._bbox_is_centered(p, img_w, img_h):
                    w *= 1.3
                all_hits.append((cls, conf * w))

        if not all_hits:
            return None, None, 0

        scores = defaultdict(float)
        counts = defaultdict(int)
        for cls, score in all_hits:
            scores[cls] += score
            counts[cls] += 1

        # merge cap+can co-occurrences (see docstring on the method)
        scores, counts = self._merge_cooccurrences(scores, counts)

        if DEBUG:
            print("  Scores:")
            for cls in sorted(scores, key=scores.get, reverse=True):
                pct = self._score_to_pct(scores[cls])
                print(f"    {cls:30s}  {scores[cls]:5.2f} ({pct}%)  "
                      f"hits={counts[cls]}  -> {self._get_bin(cls)}")

        bin_name, display_label = self._pick_best_label(scores)
        raw_score = scores[display_label]
        pct = self._score_to_pct(raw_score)

        print(f"  Result: {display_label} -> {bin_name} ({pct}%)")
        return bin_name, display_label, pct

    # --- UI drawing ---

    def _draw_overlay(self, frame, bbox, auto_pct, light_warn):
        out = frame.copy()
        h, w = out.shape[:2]
        mx = int(w * 0.10)
        my = int(h * 0.10)
        x1, y1 = mx, my
        x2, y2 = w - mx, h - my

        # dim area outside the guide box
        mask = np.zeros_like(out, dtype=np.uint8)
        mask[y1:y2, x1:x2] = out[y1:y2, x1:x2]
        out = cv2.addWeighted(out, 0.35, mask, 0.65, 0)
        out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        # corner brackets
        color = (0, 255, 255)
        arm = 40
        for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = arm if cx == x1 else -arm
            dy = arm if cy == y1 else -arm
            cv2.line(out, (cx, cy), (cx + dx, cy), color, 3)
            cv2.line(out, (cx, cy), (cx, cy + dy), color, 3)

        # green box around detected object
        if bbox:
            bx, by, bw, bh = bbox
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        # top bar
        cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(out, "EcoSort G16",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if light_warn:
            cv2.putText(out, f"WARNING: {light_warn}",
                        (w // 2 - 120, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 100, 255), 2)

        # mini last result in top right (shows for 30s after a scan)
        now = time.time()
        if self._last_result_bin and now - self._last_result_time < 30:
            res_color = BIN_COLORS_BGR.get(self._last_result_bin, (200, 200, 200))
            info = BIN_DISPLAY.get(self._last_result_bin,
                                   {"label": self._last_result_bin})
            mini_text = (f"{info['label']}: {self._last_result_label} "
                         f"{self._last_result_score}%")
            (tw, th), _ = cv2.getTextSize(
                mini_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = w - tw - 12
            cv2.rectangle(out, (tx - 6, 8), (w - 6, 32), (40, 40, 40), -1)
            cv2.putText(out, mini_text,
                        (tx, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, res_color, 1)
        else:
            cv2.putText(out, f"Scans: {self.total_scans}",
                        (w - 140, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

        # prompt if background hasnt been captured yet
        if not self.bg_ready:
            prompt1 = "Clear the area and press B"
            prompt2 = "to capture background"
            (tw1, _), _ = cv2.getTextSize(
                prompt1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            (tw2, _), _ = cv2.getTextSize(
                prompt2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(out, (w // 2 - 250, h // 2 - 60),
                          (w // 2 + 250, h // 2 + 50), (0, 0, 0), -1)
            cv2.putText(out, prompt1,
                        (w // 2 - tw1 // 2, h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.putText(out, prompt2,
                        (w // 2 - tw2 // 2, h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        # auto-scan progress bar
        if auto_pct > 0 and bbox and self.bg_ready:
            bar_y = y2 + 12
            bar_w = int((x2 - x1) * auto_pct)
            cv2.rectangle(out, (x1, bar_y), (x2, bar_y + 8), (60, 60, 60), -1)
            bar_color = (0, 255, 0) if auto_pct < 0.9 else (0, 255, 255)
            cv2.rectangle(out, (x1, bar_y), (x1 + bar_w, bar_y + 8),
                          bar_color, -1)
            cv2.putText(out, "Auto-scan",
                        (x1, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (180, 180, 180), 1)

        # bottom bar with controls
        cv2.rectangle(out, (0, h - 32), (w, h), (0, 0, 0), -1)
        bg_status = "BG set" if self.bg_ready else "NO BG - press B"
        cv2.putText(out,
                    f"SPACE=scan | B=background | S=stats | F=fullscreen | Q=quit  [{bg_status}]",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 180, 180), 1)

        return out

    def _draw_result_screen(self, frame, bin_name, label, pct):
        out = frame.copy()
        h, w = out.shape[:2]
        color = BIN_COLORS_BGR.get(bin_name, (255, 255, 255))
        info = BIN_DISPLAY.get(bin_name, {"icon": "?", "label": bin_name})

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.85, out, 0.15, 0)
        out[130:] = cv2.addWeighted(frame[130:], 0.6,
                                    np.zeros_like(frame[130:]), 0.4, 0)

        cv2.rectangle(out, (0, 0), (w, 6), color, -1)

        cv2.circle(out, (60, 68), 38, color, -1)
        cv2.putText(out, info["icon"],
                    (43, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

        cv2.putText(out, f"BIN: {info['label']}",
                    (115, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        cv2.putText(out, f"Detected: {label}",
                    (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # confidence bar - green if good, orange if ok, red if bad
        conf_color = (0, 200, 0) if pct >= 70 else (
            (0, 200, 255) if pct >= 40 else (0, 100, 255))
        cv2.putText(out, f"Confidence: {pct}%",
                    (115, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conf_color, 2)

        bar_x, bar_y = 310, 96
        bar_max_w = 200
        bar_w = int(bar_max_w * pct / 100)
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + 14),
                      (60, 60, 60), -1)
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14),
                      conf_color, -1)

        return out

    def _draw_stats_screen(self, frame):
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, h), (20, 20, 20), -1)

        cv2.putText(out, "EcoSort G16 - Session Stats",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(out, f"Total scans: {self.total_scans}   Unsure: {self.total_unsure}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.line(out, (30, 110), (w - 30, 110), (80, 80, 80), 1)

        bins_ordered = ["RETURNABLES", "RECYCLING", "COMPOST", "GARBAGE", "E-WASTE"]
        max_count = max((self.stats.get(b, 0) for b in bins_ordered), default=1) or 1
        bar_area_w = w - 260

        for i, bname in enumerate(bins_ordered):
            count = self.stats.get(bname, 0)
            color = BIN_COLORS_BGR.get(bname, (180, 180, 180))
            info = BIN_DISPLAY.get(bname, {"label": bname})
            y = 140 + i * 70

            cv2.putText(out, info["label"],
                        (30, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            bar_w = int((count / max_count) * bar_area_w) if count > 0 else 4
            cv2.rectangle(out, (220, y + 5), (220 + bar_w, y + 35), color, -1)
            cv2.putText(out, str(count),
                        (230 + bar_w, y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        cv2.putText(out, "Press any key to return",
                    (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        return out

    def _draw_unsure_screen(self, frame, pct):
        out = frame.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.rectangle(out, (0, 0), (w, 6), (0, 100, 255), -1)
        cv2.putText(out, "UNSURE - try again",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 100, 255), 3)
        cv2.putText(out, f"Confidence {pct}% is below threshold {CONFIDENCE_FLOOR}%",
                    (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return out

    def _show(self, frame):
        if not HEADLESS:
            cv2.imshow(self._window_name, frame)

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            cv2.setWindowProperty(self._window_name,
                                  cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self._window_name,
                                  cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    def _do_scan(self, frame):
        ts = time.strftime("%H:%M:%S")
        print(f"\n[{ts}] Scanning...")
        lcd_show("Scanning...", "Hold still")
        led_off()

        scanning = frame.copy()
        cv2.rectangle(scanning, (0, 0), (CAPTURE_W, 90), (0, 0, 0), -1)
        cv2.rectangle(scanning, (0, 0), (CAPTURE_W, 6), (0, 255, 255), -1)
        cv2.putText(scanning, "Scanning...",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        self._show(scanning)
        cv2.waitKey(1)

        bin_name, label, pct = self._infer(frame)
        self.total_scans += 1

        if bin_name and pct >= CONFIDENCE_FLOOR:
            self.stats[bin_name] += 1
            self._consecutive_no_det = 0

            self._last_result_bin = bin_name
            self._last_result_label = label
            self._last_result_score = pct
            self._last_result_time = time.time()

            self._show(self._draw_result_screen(frame, bin_name, label, pct))
            lcd_show(f"Bin: {bin_name}", f"Found: {label}")
            led_blink(bin_name)

            if not HEADLESS:
                cv2.waitKey(RESULT_DISPLAY)
            led_off()
            lcd_show("EcoSort Ready", "SPACE to scan")
            return True

        elif bin_name and pct < CONFIDENCE_FLOOR:
            self.total_unsure += 1
            self._consecutive_no_det += 1
            print(f"[{ts}] Unsure ({pct}% < {CONFIDENCE_FLOOR}%)")

            self._show(self._draw_unsure_screen(frame, pct))
            lcd_show("Unsure result", f"Confidence {pct}%")
            led_off()

            if not HEADLESS:
                cv2.waitKey(NODET_DISPLAY)
            lcd_show("EcoSort Ready", "SPACE to scan")

        else:
            self._consecutive_no_det += 1
            print(f"[{ts}] Nothing detected (streak: {self._consecutive_no_det})")

            no_det = frame.copy()
            cv2.rectangle(no_det, (0, 0), (CAPTURE_W, 90), (0, 0, 0), -1)
            cv2.putText(no_det, "Nothing detected - try again",
                        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 3)
            self._show(no_det)
            lcd_show("No detection", "Try again")
            led_off()

            if not HEADLESS:
                cv2.waitKey(NODET_DISPLAY)
            lcd_show("EcoSort Ready", "SPACE to scan")

        # if we keep failing, the background probably drifted
        if self._consecutive_no_det >= AUTO_BG_RECAPTURE_AFTER:
            print(f"  {self._consecutive_no_det} failures in a row, re-capturing background")
            self._capture_background()
            lcd_show("EcoSort Ready", "SPACE to scan")

        return False

    # main loop

    def run(self):
        self._open_camera()

        if not HEADLESS:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._window_name, CAPTURE_W, CAPTURE_H)

        print("\nEcoSort G16 ready.")
        print("  Press B to capture background (required before scanning)")
        print("  SPACE = scan | S = stats | F = fullscreen | Q = quit\n")
        lcd_show("Press B to", "set background")

        try:
            while True:
                frame = self._read_frame()

                bbox = None
                has_object = False

                if self.bg_ready:
                    bbox = self._find_object_bbox(frame)
                    has_object = bbox is not None
                    self._adapt_background(frame, has_object)

                light_warn = check_lighting(frame)

                auto_pct = 0.0
                if has_object and self.bg_ready:
                    auto_pct = min(1.0, self._stable_count / AUTO_SCAN_STABLE_FRAMES)

                self._show(self._draw_overlay(frame, bbox, auto_pct, light_warn))

                if self.bg_ready and self._check_auto_scan(bbox):
                    print("  Auto-scan triggered!")
                    self._do_scan(frame)
                    self._reset_auto_scan()
                    continue

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Shutting down...")
                    lcd_show("Shutting down...")
                    break
                elif key == ord("b"):
                    self._capture_background()
                    self._reset_auto_scan()
                    lcd_show("EcoSort Ready", "SPACE to scan")
                elif key == ord("f"):
                    self._toggle_fullscreen()
                elif key == ord("s"):
                    self._show(self._draw_stats_screen(frame))
                    cv2.waitKey(0)
                elif key == 32:  # spacebar
                    if not self.bg_ready:
                        print("Background not set! Press B first.")
                        lcd_show("Press B first!", "Set background")
                        time.sleep(1.5)
                        lcd_show("Press B to", "set background")
                    else:
                        self._do_scan(frame)
                        self._reset_auto_scan()

        finally:
            _stop_scroll()
            self._close_camera()
            led_cleanup()
            lcd_show("Goodbye!", "")
            if not HEADLESS:
                cv2.destroyAllWindows()

            # print session summary
            print("\nSession Stats")
            print(f"  Total scans:  {self.total_scans}")
            print(f"  Unsure:       {self.total_unsure}")
            for b in ["RETURNABLES", "RECYCLING", "COMPOST", "GARBAGE", "E-WASTE"]:
                print(f"  {b:14s}: {self.stats.get(b, 0)}")


if __name__ == "__main__":
    scanner = EcoSortScanner()
    scanner.run()