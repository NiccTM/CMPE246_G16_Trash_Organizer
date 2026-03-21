# EcoSort G16: Automated Waste Classification System
### CMPE 246 - Design Studio | University of British Columbia

## 1. Project Overview
**EcoSort G16** is an intelligent waste management solution designed to automate the classification of refuse into appropriate disposal streams. Using a **Raspberry Pi** and a **YOLOv8** computer vision model, the system identifies items in real-time and provides high-visibility UI feedback to the user, ensuring higher recycling accuracy and reduced contamination.



---

## 2. System Architecture
The project is built on a modular architecture that bridges cloud-based AI with local hardware execution.

### Hardware Stack
* **Processor:** Raspberry Pi 4 Model B
* **Vision:** Raspberry Pi Camera Module V2
* **Connectivity:** SSH (Secure Shell) for remote command execution and headless management.

### Software Stack
* **Language:** Python 3.10
* **Libraries:** `OpenCV`, `NumPy`, `Roboflow`
* **Model:** YOLOv8 (Trained on TACO V11 Dataset)

---

## 3. Logic Mapping & UI Design
The system translates 12 distinct AI-detected classes into 4 physical bin destinations. Each destination is assigned a high-contrast neon color to assist users in low-light environments.

| Target Bin | AI Detection Classes | UI Color (BGR) |
| :--- | :--- | :--- |
| **FOOD SCRAPS** | Food, Organic waste | (0, 255, 0) - Lime |
| **RECYCLABLES** | Cans, Glass, Plastic containers, Lids | (255, 255, 0) - Cyan |
| **PAPER** | Paper, Cardboard | (255, 150, 50) - Blue |
| **GARBAGE** | Trash, Styrofoam, Utensils, Plastic bags | (255, 0, 255) - Magenta |

---

## 4. Setup and Installation

### Remote Access (Lab 7/8 Milestone)
To comply with the project guidelines for remote management, ensure SSH is enabled on the Pi:
1. Run `sudo raspi-config`.
2. Navigate to **Interfacing Options** > **SSH** > **Enable**.

### Environment Setup
Install the necessary dependencies on the Raspberry Pi:

`pip install opencv-python numpy roboflow`

# Presentation Information

