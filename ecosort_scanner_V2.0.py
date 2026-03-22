import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

"""
EcoSort G16 - Raspberry Pi 5 + Freenove 8MP (IMX219) + 1602 I2C LCD
=====================================================================
Bin categories matched to UBC "Sort it Out" campus bins:
  Garbage | Recycling | Returnables | Compost | E-Waste

HOW TO USE:
  1. Start the app with NO object in frame (just your desk/surface).
     It captures a background reference automatically.
  2. Place an object in the center of the frame.
  3. Press SPACE to scan.
  4. Press B at any time to re-capture the background.

HARDWARE SETUP:
  Camera:
    1. Power off Pi, insert ribbon cable into CAM0 (blue stripe faces AWAY)
    2. sudo nano /boot/firmware/config.txt
       Add: camera_auto_detect=0
            dtoverlay=imx219,cam0
    3. sudo reboot
    4. Verify: rpicam-hello --list-cameras  (should show imx219)

  1602 I2C LCD (PCF8574 backpack):
    LCD GND → Pi Pin 6  (GND)
    LCD VCC → Pi Pin 2  (5V)
    LCD SDA → Pi Pin 3  (GPIO 2 / SDA1)
    LCD SCL → Pi Pin 5  (GPIO 3 / SCL1)

  Roboflow:
    Create .env file next to this script: ROBOFLOW_API_KEY=your_key_here
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

# ── LCD setup (graceful fallback if not connected) ────────────────────────────
LCD_AVAILABLE = False
lcd = None

LCD_I2C_ADDR  = 0x27
LCD_COLS      = 16
LCD_ROWS      = 2

try:
    from RPLCD.i2c import CharLCD
    lcd = CharLCD(i2c_expander='PCF8574', address=LCD_I2C_ADDR,
                  port=1, cols=LCD_COLS, rows=LCD_ROWS)
    lcd.clear()
    LCD_AVAILABLE = True
    print(f"LCD connected at 0x{LCD_I2C_ADDR:02x}")
except Exception as e:
    print(f"LCD not available ({e}) — running without LCD")

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_PORT    = 0
CAPTURE_W      = 1280
CAPTURE_H      = 720
CONFIDENCE     = 15
SNAP_PATH      = "/tmp/ecosort_snap.jpg"
HEADLESS       = False
DEBUG          = True
RESULT_DISPLAY = 8000
NODET_DISPLAY  = 4000
LCD_SCROLL_DELAY = 0.35

# Background subtraction settings
BG_DIFF_THRESH   = 35    # Pixel diff threshold to count as "changed"
MIN_OBJECT_AREA  = 3000  # Minimum contour area (pixels²) to be a real object
OBJECT_PAD       = 60    # Extra padding (px) around detected object bbox
# ─────────────────────────────────────────────────────────────────────────────

SPECIFIC_CLASSES = {
    "can", "aluminum", "bottle", "glass", "plastic container",
    "cap or lid", "paper", "cardboard boxes", "cardboard boxes and cartons",
    "battery", "styrofoam", "food & others",
}

BACKGROUND_PRONE = {
    "plastic bag or wrapper", "plastic bag", "utensils and straw",
    "utensils/straw", "trash",
}

SAME_BIN_TIE_THRESHOLD = 0.15

LABEL_SPECIFICITY = {
    "can": 10, "aluminum": 10, "bottle": 10, "glass": 10,
    "battery": 10, "styrofoam": 10,
    "plastic container": 8,
    "paper": 7, "cardboard boxes": 7, "cardboard boxes and cartons": 7,
    "food & others": 7,
    "cap or lid": 3,
    "container": 3,
    "trash": 1,
    "plastic bag or wrapper": 1,
    "utensils and straw": 1,
}

BIN_COLORS = {
    "GARBAGE":      (128, 128, 128),
    "RECYCLING":    (255, 180, 0),
    "RETURNABLES":  (0, 180, 0),
    "COMPOST":      (0, 165, 255),
    "E-WASTE":      (180, 105, 255),
}


# ── LCD helpers ───────────────────────────────────────────────────────────────

_scroll_thread = None
_scroll_stop = threading.Event()


def lcd_show(line1: str, line2: str = ""):
    if not LCD_AVAILABLE:
        return
    _stop_scroll()
    if len(line1) > LCD_COLS or len(line2) > LCD_COLS:
        _start_scroll(line1, line2)
    else:
        _lcd_write_static(line1, line2)


def _lcd_write_static(line1: str, line2: str = ""):
    try:
        lcd.clear()
        lcd.cursor_pos = (0, 0)
        lcd.write_string(line1[:LCD_COLS].ljust(LCD_COLS))
        if line2:
            lcd.cursor_pos = (1, 0)
            lcd.write_string(line2[:LCD_COLS].ljust(LCD_COLS))
    except Exception as e:
        print(f"  LCD write error: {e}")


def _scroll_worker(line1: str, line2: str):
    pad = " " * 3
    text1 = (line1 + pad + line1) if len(line1) > LCD_COLS else line1.ljust(LCD_COLS)
    text2 = (line2 + pad + line2) if len(line2) > LCD_COLS else line2.ljust(LCD_COLS)
    scroll1 = len(line1) > LCD_COLS
    scroll2 = len(line2) > LCD_COLS
    _lcd_write_static(line1, line2)
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
                window = "".join(text1[(offset1 + i) % len(text1)]
                                 for i in range(LCD_COLS))
                lcd.write_string(window)
            else:
                lcd.write_string(text1[:LCD_COLS])
            lcd.cursor_pos = (1, 0)
            if scroll2:
                window = "".join(text2[(offset2 + i) % len(text2)]
                                 for i in range(LCD_COLS))
                lcd.write_string(window)
            else:
                lcd.write_string(text2[:LCD_COLS])
        except Exception:
            pass
        if scroll1:
            offset1 = (offset1 + 1) % wrap1
        if scroll2:
            offset2 = (offset2 + 1) % wrap2
        time.sleep(LCD_SCROLL_DELAY)


def _start_scroll(line1: str, line2: str):
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
        self.bg_frame = None  # Background reference (grayscale, blurred)

    def _get_bin(self, label: str) -> str:
        label_lower = label.lower()
        for bin_name, keywords in self.bin_logic.items():
            if any(k in label_lower for k in keywords):
                return bin_name
        return "GARBAGE"

    # ── Camera ────────────────────────────────────────────────────────────────

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

    def _read_frame(self) -> np.ndarray:
        raw = self.camera.capture_array()
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def _close_camera(self):
        if self.camera:
            self.camera.stop()
            print("Camera released.")

    # ── Background subtraction ────────────────────────────────────────────────

    def _capture_background(self):
        """Capture multiple frames and average them for a stable background."""
        print("Capturing background (keep scene empty)...")
        lcd_show("Capturing BG...", "Keep area clear")

        frames = []
        for _ in range(10):
            frames.append(self._read_frame())
            time.sleep(0.1)

        # Average the frames to reduce noise
        avg = np.mean(frames, axis=0).astype(np.uint8)
        self.bg_frame = cv2.GaussianBlur(
            cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        print("Background captured.")
        lcd_show("BG captured!", "Place object")
        time.sleep(1.0)

    def _find_object_bbox(self, frame: np.ndarray):
        """
        Compare current frame to background, find the largest new object.
        Returns (x, y, w, h) bounding box or None if nothing found.
        """
        if self.bg_frame is None:
            return None

        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        # Absolute difference from background
        diff = cv2.absdiff(self.bg_frame, gray)
        _, thresh = cv2.threshold(diff, BG_DIFF_THRESH, 255, cv2.THRESH_BINARY)

        # Clean up noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Take the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_OBJECT_AREA:
            if DEBUG:
                print(f"  Object too small ({area} px²), min={MIN_OBJECT_AREA}")
            return None

        x, y, w, h = cv2.boundingRect(largest)

        # Add padding and clamp to frame bounds
        fh, fw = frame.shape[:2]
        x1 = max(0, x - OBJECT_PAD)
        y1 = max(0, y - OBJECT_PAD)
        x2 = min(fw, x + w + OBJECT_PAD)
        y2 = min(fh, y + h + OBJECT_PAD)

        if DEBUG:
            print(f"  Object found: {x2-x1}x{y2-y1} px at ({x1},{y1}), "
                  f"area={area}")

        return (x1, y1, x2 - x1, y2 - y1)

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _bbox_is_centered(pred: dict, img_w: int, img_h: int,
                          tolerance: float = 0.40) -> bool:
        cx = pred["x"] / img_w
        cy = pred["y"] / img_h
        return abs(cx - 0.5) < tolerance and abs(cy - 0.5) < tolerance

    # ── Same-bin tie-breaking ─────────────────────────────────────────────────

    def _pick_best_label(self, scores: dict[str, float]) -> tuple[str, str]:
        ranked = sorted(scores, key=scores.get, reverse=True)
        winner = ranked[0]
        winner_bin = self._get_bin(winner)
        if len(ranked) >= 2:
            runner_up = ranked[1]
            runner_up_bin = self._get_bin(runner_up)
            if (runner_up_bin == winner_bin
                    and scores[winner] > 0
                    and (scores[winner] - scores[runner_up]) / scores[winner]
                    <= SAME_BIN_TIE_THRESHOLD):
                w_spec = LABEL_SPECIFICITY.get(winner.lower(), 5)
                r_spec = LABEL_SPECIFICITY.get(runner_up.lower(), 5)
                if r_spec > w_spec:
                    if DEBUG:
                        print(f"  ↳ Tie-break: '{runner_up}' "
                              f"(spec {r_spec}) over '{winner}' "
                              f"(spec {w_spec})")
                    return winner_bin, runner_up
        return winner_bin, winner

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray):
        """
        1. Use background subtraction to find the object's bounding box.
        2. Crop to the object (+ padding), isolating it from background.
        3. Run inference on the isolated crop in 3 variants:
           - raw crop
           - CLAHE-enhanced crop
           - slightly padded crop (more context)
        4. Score and pick the best class.
        """
        bbox = self._find_object_bbox(frame)

        if bbox is None:
            # Fallback: use center crop if background subtraction fails
            print("  BG subtraction found nothing, using center crop fallback")
            h, w = frame.shape[:2]
            margin = 0.20
            mx, my = int(w * margin), int(h * margin)
            bbox = (mx, my, w - 2 * mx, h - 2 * my)

        bx, by, bw, bh = bbox
        object_crop = frame[by:by+bh, bx:bx+bw]

        # Build inference variants from the isolated object
        variants = [
            ("isolated",     object_crop),
            ("enhanced",     self._enhance(object_crop)),
        ]

        # Also try a slightly wider crop for more context
        fh, fw = frame.shape[:2]
        extra = OBJECT_PAD
        wx1 = max(0, bx - extra)
        wy1 = max(0, by - extra)
        wx2 = min(fw, bx + bw + extra)
        wy2 = min(fh, by + bh + extra)
        wider_crop = frame[wy1:wy2, wx1:wx2]
        variants.append(("wider", wider_crop))

        all_hits: list[tuple[str, float]] = []

        for var_name, img in variants:
            img_h, img_w = img.shape[:2]
            cv2.imwrite(SNAP_PATH, img)
            try:
                results = self.model.predict(
                    SNAP_PATH, confidence=CONFIDENCE
                ).json()
                preds = results.get("predictions", [])
            except Exception as e:
                print(f"    [{var_name}] error: {e}")
                continue

            if DEBUG:
                brief = [(p["class"], round(p["confidence"], 2))
                         for p in preds]
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
            return None, None

        scores: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for cls, score in all_hits:
            scores[cls] += score
            counts[cls] += 1

        if DEBUG:
            print("  ── Aggregated scores ──")
            for cls in sorted(scores, key=scores.get, reverse=True):
                print(f"    {cls:30s}  score={scores[cls]:6.2f}  "
                      f"hits={counts[cls]}  → {self._get_bin(cls)}")

        bin_name, display_label = self._pick_best_label(scores)
        print(f"  ✓ Result: {display_label} → {bin_name} "
              f"(score={scores[display_label]:.2f}, "
              f"hits={counts[display_label]})")

        return bin_name, display_label

    # ── Display helpers ───────────────────────────────────────────────────────

    def _show(self, title: str, frame: np.ndarray):
        if not HEADLESS:
            cv2.imshow(title, frame)

    def _draw_guide(self, frame: np.ndarray) -> np.ndarray:
        """Draw the guide box, and if BG subtraction finds an object,
        draw a green box around it in real time."""
        out = frame.copy()
        h, w = out.shape[:2]
        mx = int(w * 0.10)
        my = int(h * 0.10)
        x1, y1 = mx, my
        x2, y2 = w - mx, h - my

        # Dim outside
        mask = np.zeros_like(out, dtype=np.uint8)
        mask[y1:y2, x1:x2] = out[y1:y2, x1:x2]
        out = cv2.addWeighted(out, 0.35, mask, 0.65, 0)
        out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        # Corner brackets
        color = (0, 255, 255)
        arm = 40
        for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = arm if cx == x1 else -arm
            dy = arm if cy == y1 else -arm
            cv2.line(out, (cx, cy), (cx + dx, cy), color, 3)
            cv2.line(out, (cx, cy), (cx, cy + dy), color, 3)

        # Live object detection box (green rectangle)
        bbox = self._find_object_bbox(frame)
        if bbox:
            bx, by, bw, bh = bbox
            cv2.rectangle(out, (bx, by), (bx + bw, by + bh),
                          (0, 255, 0), 2)
            cv2.putText(out, "Object detected",
                        (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        cv2.putText(out, "Hold object inside box",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        bg_status = "BG set" if self.bg_frame is not None else "NO BG"
        cv2.putText(out, f"SPACE=scan | B=set background | Q=quit  [{bg_status}]",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1)
        return out

    def _draw_result(self, frame: np.ndarray, bin_name: str,
                     label: str) -> np.ndarray:
        color = BIN_COLORS.get(bin_name, (255, 255, 255))
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (CAPTURE_W, 90), (0, 0, 0), -1)
        cv2.putText(out, f"BIN: {bin_name}",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
        cv2.putText(out, f"Detected: {label}",
                    (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1)
        return out

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self._open_camera()

        # Auto-capture background on startup
        self._capture_background()

        print("\nEcoSort G16 ready. SPACE=scan, B=set background, Q=quit\n")
        lcd_show("EcoSort Ready", "SPACE to scan")

        try:
            while True:
                frame = self._read_frame()
                self._show("EcoSort G16", self._draw_guide(frame))

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Shutting down...")
                    lcd_show("Shutting down...")
                    break

                elif key == ord("b"):
                    self._capture_background()
                    lcd_show("EcoSort Ready", "SPACE to scan")

                elif key == 32:
                    ts = time.strftime("%H:%M:%S")
                    print(f"\n[{ts}] Scanning...")

                    lcd_show("Scanning...", "Hold still")

                    scanning_frame = frame.copy()
                    cv2.rectangle(scanning_frame, (0, 0),
                                  (CAPTURE_W, 90), (0, 0, 0), -1)
                    cv2.putText(scanning_frame, "Scanning...",
                                (15, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                1.6, (0, 255, 255), 4)
                    self._show("EcoSort G16", scanning_frame)
                    cv2.waitKey(1)

                    bin_name, label = self._infer(frame)

                    if bin_name:
                        self._show("EcoSort G16",
                                   self._draw_result(frame, bin_name, label))

                        lcd_show(
                            f"Bin: {bin_name}",
                            f"Found: {label}"
                        )

                        if not HEADLESS:
                            cv2.waitKey(RESULT_DISPLAY)

                        lcd_show("EcoSort Ready", "SPACE to scan")

                    else:
                        print(f"[{ts}] Nothing detected - try again.")
                        lcd_show("No detection", "Try again")

                        no_det = frame.copy()
                        cv2.rectangle(no_det, (0, 0),
                                      (CAPTURE_W, 90), (0, 0, 0), -1)
                        cv2.putText(no_det,
                                    "Nothing detected - try again",
                                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 100, 255), 3)
                        self._show("EcoSort G16", no_det)
                        if not HEADLESS:
                            cv2.waitKey(NODET_DISPLAY)

                        lcd_show("EcoSort Ready", "SPACE to scan")

        finally:
            _stop_scroll()
            self._close_camera()
            lcd_show("Goodbye!", "")
            if not HEADLESS:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    scanner = EcoSortScanner()
    scanner.run()