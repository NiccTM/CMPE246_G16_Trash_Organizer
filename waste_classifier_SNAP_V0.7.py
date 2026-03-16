import cv2
import numpy as np
import os
from roboflow import Roboflow

class EcoSortScanner:
    def __init__(self, api_key):
        # Initialize the TACO V11 model from Roboflow Cloud
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("taco-ihjgk").project("yolov8-trash-detections-kgnug")
        self.model = project.version(11).model
        
        # Hardware Mapping: 12 AI classes -> 4 physical bins
        self.bin_logic = {
            "FOOD SCRAPS": ["food & others"],
            "RECYCLABLE CONTAINERS": ["can", "aluminum", "glass", "plastic container", "cap or lid"],
            "PAPER": ["paper", "cardboard boxes"],
            "GARBAGE": ["trash", "battery", "styrofoam", "plastic bag", "utensils/straw"]
        }

    def _get_bin(self, label):
        """Routes recognized items to the correct physical bin, defaulting to Garbage."""
        label = label.lower()
        for b_name, materials in self.bin_logic.items():
            if label in materials:
                return b_name
        return "GARBAGE"

    def _draw_ui(self, img, pred):
        """Draws bounding boxes and multi-line labels using high-visibility neon colors."""
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        target_bin = self._get_bin(pred['class'])
        
        # High-Visibility UI Palette (BGR Format)
        color_map = {
            "FOOD SCRAPS": (0, 255, 0),             # Neon Lime
            "RECYCLABLE CONTAINERS": (255, 255, 0), # Bright Cyan
            "PAPER": (255, 150, 50),                # Sky Blue
            "GARBAGE": (255, 0, 255)                # Magenta
        }
        color = color_map.get(target_bin, (255, 255, 255))
        
        # Draw bold bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Split text into two lines to prevent overlap
        label_line1 = f"{pred['class'].upper()} ({pred['confidence']:.0%})"
        label_line2 = f"-> {target_bin}"
        label_y1 = max(20, y1 - 25) 
        label_y2 = label_y1 + 20 
        
        cv2.putText(img, label_line1, (x1, label_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, label_line2, (x1, label_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return img, target_bin

    def run_scanner(self):
        """Main camera loop handling live feed, capture, and API inference."""
        cap = cv2.VideoCapture(0)
        print("\n" + "="*55)
        print("EcoSort G16: High-Visibility UI Active")
        print(" -> Screen Colors: Lime (Food) | Cyan (Containers)")
        print("                   Sky Blue (Paper) | Magenta (Garbage)")
        print("="*55 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. Live Viewfinder State
            preview = frame.copy()
            cv2.putText(preview, "READY: Press SPACE to Scan", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("EcoSort G16 - Viewfinder", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # 2. Capture & Analyze State
            elif key == 32: 
                # Shutter flash effect
                cv2.imshow("EcoSort G16 - Viewfinder", np.zeros_like(frame))
                cv2.waitKey(50)
                
                temp_path = "current_snap.jpg"
                cv2.imwrite(temp_path, frame) 
                
                snap_frame = frame.copy()
                cv2.putText(snap_frame, "ANALYZING...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("EcoSort G16 - Viewfinder", snap_frame)
                cv2.waitKey(1)

                # API Call (25% confidence threshold for TACO V11)
                raw_response = self.model.predict(temp_path, confidence=25)
                prediction_result = raw_response.json()
                preds = prediction_result.get('predictions', [])
                result_img = cv2.imread(temp_path)
                
                # 3. Result Display State
                if preds:
                    primary_bin = "GARBAGE"
                    for idx, p in enumerate(preds):
                        result_img, b_name = self._draw_ui(result_img, p)
                        if idx == 0: 
                            primary_bin = b_name

                    # Draw the final instruction banner
                    h, w, _ = result_img.shape
                    cv2.rectangle(result_img, (0, h-70), (w, h), (0, 0, 0), -1)
                    banner_text = f"DESTINATION: {primary_bin}"
                    cv2.putText(result_img, banner_text, (20, h-25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                    cv2.imshow("Scan Result", result_img)
                    cv2.waitKey(0)  
                    cv2.destroyWindow("Scan Result")
                else:
                    print("Detection failed. Reposition the item and try again.")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nScanner shut down safely.")

if __name__ == "__main__":
    MY_KEY = "DcJDGFd9CDdi4yOTljMJ"
    scanner = EcoSortScanner(api_key=MY_KEY)
    scanner.run_scanner()