import cv2
import numpy as np
import os
from roboflow import Roboflow

class EcoSortScanner:
    def __init__(self, api_key):
        rf = Roboflow(api_key=api_key)
        # Using the TACO workspace and project
        project = rf.workspace("taco-ihjgk").project("yolov8-trash-detections-kgnug")
        
        # Confirmed Version 11 from your screenshot
        self.model = project.version(11).model
        
        # TACO 12-class mapping
        self.bin_logic = {
            "RECYCLE": ["paper", "glass", "can", "aluminum", "cardboard boxes", "plastic container"],
            "COMPOST": ["food & others"],
            "LANDFILL": ["battery", "styrofoam", "cap or lid", "plastic bag", "utensils/straw"]
        }

    def _get_bin(self, label):
        """Routes TACO's 12 classes into our 3 physical hardware bins."""
        label = label.lower()
        for b_name, materials in self.bin_logic.items():
            if label in materials:
                return b_name
        return "LANDFILL"

    def _draw_ui(self, img, pred):
        """Draws bounding boxes and individual item labels."""
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        target_bin = self._get_bin(pred['class'])
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_str = f"{pred['class'].upper()} ({pred['confidence']:.0%}) -> {target_bin}"
        cv2.putText(img, label_str, (x1, max(20, y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return img, target_bin

    def run_scanner(self):
        cap = cv2.VideoCapture(0)
        print("\n" + "="*50)
        print("EcoSort G16: TACO V11 Scanner Active")
        print(" -> Press 'SPACEBAR' to Snap & Scan")
        print(" -> Press 'Q' to Quit")
        print("="*50 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            preview = frame.copy()
            cv2.putText(preview, "READY: Press SPACE to Scan", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("EcoSort G16 - Viewfinder", preview)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 32: 
                # Analysis visual feedback
                snap_frame = frame.copy()
                cv2.putText(snap_frame, "ANALYZING...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("EcoSort G16 - Viewfinder", snap_frame)
                cv2.waitKey(1) 
                
                temp_path = "current_snap.jpg"
                cv2.imwrite(temp_path, frame) 

                print("\n--- Sending to Roboflow Cloud ---")
                
                # Confidence set to 25% for better recall on aluminum cans
                raw_response = self.model.predict(temp_path, confidence=25)
                prediction_result = raw_response.json()
                
                print(f"DEBUG API RESPONSE: {prediction_result}")
                
                preds = prediction_result.get('predictions', [])
                result_img = cv2.imread(temp_path)
                
                if preds:
                    primary_bin = "LANDFILL"
                    for idx, p in enumerate(preds):
                        result_img, b_name = self._draw_ui(result_img, p)
                        if idx == 0: 
                            primary_bin = b_name

                    h, w, _ = result_img.shape
                    cv2.rectangle(result_img, (0, h-70), (w, h), (0, 0, 0), -1)
                    banner_text = f"ACTION: PLACE IN {primary_bin}"
                    cv2.putText(result_img, banner_text, (20, h-25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                    print(f"Result: {banner_text}")
                    cv2.imshow("Scan Result", result_img)
                    
                    cv2.waitKey(0) 
                    cv2.destroyWindow("Scan Result")
                else:
                    print("No items detected. Try moving the can further from the lens.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MY_KEY = "DcJDGFd9CDdi4yOTljMJ"
    scanner = EcoSortScanner(api_key=MY_KEY)
    scanner.run_scanner()