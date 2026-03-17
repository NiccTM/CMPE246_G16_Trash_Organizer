import cv2
import numpy as np
import os
from roboflow import Roboflow

class EcoSortScanner:
    """
    CMPE 246 - Design Studio Project: EcoSort G16
    Purpose: Automate waste sorting using YOLOv8 via Roboflow API.
    """
    def __init__(self, api_key):
        # 1. Model Configuration (TACO V11)
        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("taco-ihjgk").project("yolov8-trash-detections-kgnug")
            self.model = project.version(11).model
        except Exception as e:
            print(f"Error connecting to Roboflow: {e}")
            exit(1)
        
        # 2. Hardware Mapping Logic (System Architecture)
        # Maps 12 AI classes to 4 physical bin destinations
        self.bin_logic = {
            "FOOD SCRAPS": ["food & others"],
            "RECYCLABLE CONTAINERS": ["can", "aluminum", "glass", "plastic container", "cap or lid"],
            "PAPER": ["paper", "cardboard boxes"],
            "GARBAGE": ["trash", "battery", "styrofoam", "plastic bag", "utensils/straw"]
        }

        # High-Visibility UI Palette (BGR Format) - Essential for UI/UX Evaluation
        self.colors = {
            "FOOD SCRAPS": (0, 255, 0),             # Neon Lime
            "RECYCLABLE CONTAINERS": (255, 255, 0), # Bright Cyan
            "PAPER": (255, 150, 50),                # Sky Blue
            "GARBAGE": (255, 0, 255)                # Magenta
        }

    def _get_bin(self, label):
        """Routes recognized items to the correct physical bin."""
        label = label.lower()
        for b_name, materials in self.bin_logic.items():
            if any(m in label for m in materials): # Improved matching
                return b_name
        return "GARBAGE"

    def _draw_ui(self, img, pred):
        """Draws bounding boxes and labels for the Scan Result window."""
        # Convert Roboflow coords to CV2 box
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        target_bin = self._get_bin(pred['class'])
        color = self.colors.get(target_bin, (255, 255, 255))
        
        # UI Graphics
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Multi-line label for readability
        label_line1 = f"{pred['class'].upper()} ({pred['confidence']:.0%})"
        label_line2 = f"-> {target_bin}"
        
        # Dynamic text placement
        text_y = max(25, y1 - 30)
        cv2.putText(img, label_line1, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, label_line2, (x1, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return img, target_bin

    def run_scanner(self):
        """Main camera loop handling live feed and capture."""
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*55)
        print("EcoSort G16: System Active")
        print(" -> Control: [SPACE] to Scan | [Q] to Quit")
        print("="*55 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Viewfinder UI
            preview = frame.copy()
            cv2.putText(preview, "READY: SPACE to Scan", (15, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("EcoSort G16 - Viewfinder", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Action: Capture and Inference
            elif key == 32: 
                print("Scanning item...")
                
                # Visual feedback: Flash
                cv2.imshow("EcoSort G16 - Viewfinder", np.ones_like(frame)*255)
                cv2.waitKey(50)
                
                temp_path = "current_snap.jpg"
                cv2.imwrite(temp_path, frame) 

                # API Call
                try:
                    raw_response = self.model.predict(temp_path, confidence=25)
                    preds = raw_response.json().get('predictions', [])
                except Exception as e:
                    print(f"Inference failed: {e}")
                    continue

                if preds:
                    result_img = frame.copy()
                    primary_bin = self._get_bin(preds[0]['class'])
                    
                    for p in preds:
                        result_img, _ = self._draw_ui(result_img, p)

                    # Final Instruction Banner
                    h, w, _ = result_img.shape
                    cv2.rectangle(result_img, (0, h-80), (w, h), (0,0,0), -1)
                    cv2.putText(result_img, f"SORT TO: {primary_bin}", (20, h-30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

                    cv2.imshow("Scan Result - Press ANY KEY to continue", result_img)
                    cv2.waitKey(0) 
                    cv2.destroyWindow("Scan Result - Press ANY KEY to continue")
                else:
                    print("No objects detected. Try adjusting lighting or distance.")

        cap.release()
        cv2.destroyAllWindows()
        if os.path.exists("current_snap.jpg"): os.remove("current_snap.jpg")

if __name__ == "__main__":
    # Note: Keep API keys in environment variables for better engineering practice
    MY_KEY = "DcJDGFd9CDdi4yOTljMJ" 
    scanner = EcoSortScanner(api_key=MY_KEY)
    scanner.run_scanner()