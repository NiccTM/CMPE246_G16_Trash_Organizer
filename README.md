# EcoSort G16: Automated Waste Classification System

### CMPE 246 - Design Studio | University of British Columbia

## 1. Project Overview

**EcoSort G16** is an intelligent waste management solution designed to automate the classification of refuse into appropriate disposal streams. Using a **Raspberry Pi 5** and a **YOLOv8** computer vision model, the system identifies items in real-time and provides high-visibility UI feedback on both an HDMI display and a 1602 I2C LCD, ensuring higher recycling accuracy and reduced contamination.

Bin categories are matched to UBC's campus **"Sort it Out"** waste stations.

---

## 2. Features

- **Background Subtraction** — Captures a reference frame of the empty scene, then isolates only the new object for inference. Eliminates false positives from desk surfaces, cables, etc.
- **Adaptive Background** — Slowly blends new frames into the background reference to handle gradual lighting changes.
- **Auto-Scan** — Automatically scans when an object is held still for ~1.5 seconds. No keyboard needed during normal operation.
- **Co-occurrence Merging** — When the model detects both "can" and "cap or lid" in the same scan, it merges the scores since the lid is the top of the can.
- **Confidence Floor** — Displays "UNSURE" when confidence is too low, preventing incorrect bin assignments.
- **Lighting Check** — Warns when the scene is too dark or too bright for reliable detection.
- **1602 I2C LCD Output** — Displays bin results, scanning status, and scrolling text for long labels.
- **Fullscreen Overlay UI** — Polished HDMI display with auto-scan progress bar, live object bounding box, confidence percentage, and mini last-result indicator.
- **Stats Screen** — Press S to view a session summary with a bar chart of items per bin.
- **Auto Background Re-capture** — Automatically re-captures background after consecutive detection failures.

---

## 3. System Architecture

### Hardware Stack

| Component | Model |
|---|---|
| Processor | Raspberry Pi 5 |
| Camera | Freenove 8MP (IMX219) on CAM0 |
| Display | HDMI monitor + 1602 I2C LCD (PCF8574) |

### Software Stack

| Component | Details |
|---|---|
| Language | Python 3.11+ |
| Libraries | OpenCV, NumPy, Roboflow, RPLCD, python-smbus |
| Model | YOLOv8 (TACO V11 Dataset via Roboflow) |

---

## 4. UBC Bin Mapping

Bin categories match the UBC campus **"Sort it Out"** waste stations:

| Target Bin | AI Detection Classes | UI Color |
|---|---|---|
| **Garbage** | Styrofoam, plastic bags/wrappers, utensils/straws, trash | Gray |
| **Recycling** | Paper, cardboard, coffee cups (cap/lid), plastic containers | Blue |
| **Returnables** | Cans, aluminum, bottles, glass | Green |
| **Compost** | Food & organic waste | Orange |
| **E-Waste** | Batteries | Pink |

---

## 5. Wiring — 1602 I2C LCD

| LCD Pin | Pi 5 Pin | GPIO Header |
|---|---|---|
| GND | GND | Pin 6 |
| VCC | 5V | Pin 2 |
| SDA | GPIO 2 (SDA1) | Pin 3 |
| SCL | GPIO 3 (SCL1) | Pin 5 |

---

## 6. Setup and Installation

### Camera Setup

1. Power off Pi, insert ribbon cable into **CAM0** (blue stripe faces AWAY from board edge).
2. Edit `/boot/firmware/config.txt`:
   ```
   camera_auto_detect=0
   dtoverlay=imx219,cam0
   ```
3. Reboot and verify: `rpicam-hello --list-cameras`

### I2C / LCD Setup

```bash
sudo raspi-config   # Interface Options → I2C → Yes
sudo reboot
sudo apt-get install -y python3-smbus i2c-tools
sudo i2cdetect -y 1  # Should show 0x27 or 0x3f
```

If your LCD shows `0x3f` instead of `0x27`, change `LCD_I2C_ADDR` in the script.

### Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### Roboflow API Key

Create a `.env` file next to the script:

```
ROBOFLOW_API_KEY=your_key_here
```

### Run

```bash
python3 ecosort_scanner_V2.0.py
```

---

## 7. Controls

| Key | Action |
|---|---|
| **B** | Capture background (required before first scan) |
| **SPACE** | Manual scan |
| **S** | Show session stats |
| **F** | Toggle fullscreen |
| **Q** | Quit |

Auto-scan triggers automatically when an object is held still for ~1.5 seconds.

---

## 8. How It Works

1. **Press B** with an empty scene to capture the background reference.
2. **Place an object** in the center of the frame — a green bounding box appears around it.
3. **Hold still** — the auto-scan progress bar fills up and scanning triggers automatically.
4. The system crops to just the detected object (removing background noise), runs inference through 3 image variants, applies weighted scoring with co-occurrence merging, and displays the result.
5. Results appear on both the HDMI overlay (with confidence %) and the LCD screen.

---

## 9. Version History

| Version | Date | Changes |
|---|---|---|
| V1.0 | 2026-03-17 | Initial release: basic Roboflow inference, 4-bin sorting, OpenCV UI |
| V2.0 | 2026-03-21 | Major rewrite: background subtraction, auto-scan, 5-bin UBC mapping, I2C LCD, adaptive background, co-occurrence merging, confidence floor, lighting check, fullscreen overlay, stats screen |

---

## About

An AI-powered waste classification system using YOLOv8 and Raspberry Pi to automate trash sorting into five streams matching UBC campus bins: Garbage, Recycling, Returnables, Compost, and E-Waste.

## License

[MIT](LICENSE)
