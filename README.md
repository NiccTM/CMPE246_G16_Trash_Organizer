# EcoSort G16 — Trash Sorting with Computer Vision

**CMPE 246 Design Studio — University of British Columbia**

## What is this?

EcoSort G16 is our group's take on making waste sorting easier. You hold an item in front of a camera, and it tells you which bin it goes in — matched to the UBC campus "Sort it Out" stations (Garbage, Recycling, Returnables, Compost, E-Waste).

It runs on a Raspberry Pi 5 with a camera module and uses a YOLOv8 model (trained on the TACO trash dataset through Roboflow) to classify what it sees. Results show up on an HDMI display and a small LCD screen.

---

## How it works

1. You start the app and press **B** to take a snapshot of the empty background (your desk, table, etc).
2. Put an item in front of the camera. The system figures out what changed in the frame (background subtraction) and crops to just the object.
3. It runs the cropped image through the model a few times with slight variations, scores the results, and picks the best match.
4. The bin name and confidence score show up on screen and on the LCD.

There's also an auto-scan feature — if you hold the object still for about 1.5 seconds, it scans automatically without needing to press anything.

---

## What we had to work around

The biggest issue was the model confusing our brown desk with cardboard. No matter how we tuned the confidence thresholds, it kept seeing "cardboard boxes" because the desk surface genuinely looks like cardboard to the model. Background subtraction was the fix — by removing the desk from the image entirely, the model only sees the actual object.

The model also kept detecting the pull-tab on top of Red Bull cans as "cap or lid" instead of "can." We added a co-occurrence rule: if both "can" and "cap or lid" show up in the same scan, we merge the lid score into "can" since it's clearly the top of the same object.

---

## Hardware

- Raspberry Pi 5
- Freenove 8MP Camera (IMX219) plugged into CAM0
- 1602 I2C LCD with PCF8574 backpack
- HDMI monitor

### LCD Wiring

| LCD Pin | Pi Pin |
|---------|--------|
| GND | Pin 6 (GND) |
| VCC | Pin 2 (5V) |
| SDA | Pin 3 (GPIO 2) |
| SCL | Pin 5 (GPIO 3) |

---

## Setup

### Camera

1. Plug the ribbon cable into CAM0 (blue stripe faces away from the board).
2. Add these lines to `/boot/firmware/config.txt`:
   ```
   camera_auto_detect=0
   dtoverlay=imx219,cam0
   ```
3. Reboot. Run `rpicam-hello --list-cameras` to check it works.

### LCD

```bash
sudo raspi-config   # Interface Options -> I2C -> Yes
sudo reboot
sudo apt-get install -y python3-smbus i2c-tools
sudo i2cdetect -y 1  # should show 0x27 or 0x3f
```

If yours shows `0x3f`, change `LCD_I2C_ADDR` at the top of the script.

### Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### API Key

Make a `.env` file next to the script:
```
ROBOFLOW_API_KEY=your_key_here
```

### Run it

```bash
python3 ecosort_scanner_V2.0.py
```

---

## Controls

| Key | What it does |
|-----|-------------|
| B | Capture background (you need to do this before scanning) |
| SPACE | Manual scan |
| S | Stats screen (shows how many items went to each bin) |
| F | Toggle fullscreen |
| Q | Quit |

Auto-scan kicks in when you hold an object still for ~1.5 seconds.

---

## Bin Categories

Matched to UBC's campus sorting stations:

| Bin | What goes in it |
|-----|----------------|
| Garbage | Styrofoam, plastic bags, straws, utensils |
| Recycling | Paper, cardboard, plastic containers, coffee cup lids |
| Returnables | Cans, bottles, glass |
| Compost | Food waste |
| E-Waste | Batteries |

---

## Other stuff worth knowing

- The background slowly adapts over time so gradual lighting changes don't break it.
- If the system gets 5 misses in a row, it re-captures the background automatically.
- If confidence is below 40%, it shows "UNSURE" instead of guessing wrong.
- It checks if the scene is too dark or too bright and shows a warning.

---

## Version History

| Version | Date | What changed |
|---------|------|-------------|
| V1.0 | Mar 17, 2026 | First version — basic detection, 4 bins, OpenCV window |
| V2.0 | Mar 21, 2026 | Background subtraction, auto-scan, 5 UBC bins, LCD support, confidence scoring, fullscreen UI, stats |

---

## License

[MIT](LICENSE)
