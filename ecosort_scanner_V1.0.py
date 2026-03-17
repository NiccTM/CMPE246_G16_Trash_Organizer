import cv2
import numpy as np
import os
from roboflow import Roboflow
# To handle security, we use environment variables
from dotenv import load_dotenv 

class EcoSortScanner:
    def __init__(self):
        # Load API key from .env file for security
        load_dotenv()
        api_key = os.getenv("ROBOFLOW_API_KEY")
        
        if not api_key:
            print("ERROR: API Key not found. Create a .env file with ROBOFLOW_API_KEY=your_key")
            exit(1)

        try:
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("taco-ihjgk").project("yolov8-trash-detections-kgnug")
            self.model = project.version(11).model
        except Exception as e:
            print(f"Connection Failed: {e}")
            exit(1)

        # Logic Mapping (System Architecture)
        self.bin_logic = {
            "FOOD SCRAPS": ["food & others"],
            "RECYCLABLES": ["can", "aluminum", "glass", "plastic container", "cap or lid"],
            "PAPER": ["paper", "cardboard boxes"],
            "GARBAGE": ["trash", "battery", "styrofoam", "plastic bag", "utensils/straw"]
        }

    def _get_bin(self, label):
        label = label.lower()
        for b_name, materials in self.bin_logic.items():
            if any(m in label for m in materials):
                return b_name
        return "GARBAGE"

    def run_scanner(self):
        cap = cv2.VideoCapture(0)
        print("\nEcoSort G16 System Initialized...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            cv2.putText(frame, "READY: SPACE to Scan", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("EcoSort G16", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == 32: # Spacebar
                temp_path = "current_snap.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Inference
                results = self.model.predict(temp_path, confidence=25).json()
                preds = results.get('predictions', [])

                if preds:
                    primary_bin = self._get_bin(preds[0]['class'])
                    print(f"Result: {preds[0]['class']} -> {primary_bin}")
                    
                    # Display Result
                    res_img = frame.copy()
                    cv2.rectangle(res_img, (0, 0), (640, 60), (0,0,0), -1)
                    cv2.putText(res_img, f"BIN: {primary_bin}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Detection", res_img)
                    cv2.waitKey(2000) # Show for 2 seconds
                    cv2.destroyWindow("Detection")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    scanner = EcoSortScanner()
    scanner.run_scanner()