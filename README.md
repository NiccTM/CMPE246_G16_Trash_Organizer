# EcoSort AI - CMPE 246 Group 16

An intelligent waste classification system using YOLOv8.

## System Features
- **Multi-Pass Analysis:** Iterates through original, sharpened, and high-contrast (CLAHE) versions of an image to maximize detection probability.
- **Waste Mapping:** Translates standard COCO dataset labels (e.g., 'vase', 'cup') into functional waste categories (Recycle, Compost, Landfill).
- **Edge Deployment Optimized:** Built using the YOLOv8 Nano model for potential Raspberry Pi integration.

## Setup Instructions
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment: `.\.venv\Scripts\activate`
4. Install dependencies: `pip install opencv-python ultralytics numpy`

## Current Status
The system successfully identifies aluminum cans (often classified as 'vase' or 'cup' due to model constraints) and maps them to the **RECYCLE** bin.