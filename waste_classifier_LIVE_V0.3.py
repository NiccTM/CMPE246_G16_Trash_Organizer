import cv2
import numpy as np
from roboflow import Roboflow
from collections import Counter

class EcoSortLive:
    def __init__(self, api_key):
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("garbage-classification-3")
        self.model = project.version(2).model
        
        self.bin_logic = {
            "RECYCLE": ["cardboard", "glass", "metal", "paper", "plastic"],
            "COMPOST": ["biodegradable"],
            "LANDFILL": ["cloth", "trash"]
        }

    def _draw_ui(self, img, pred):
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_str = f"{pred['class'].upper()} {pred['confidence']:.0%}"
        cv2.putText(img, label_str, (x1, max(20, y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    def run_live(self):
        # Open webcam (0 is usually the built-in laptop camera)
        cap = cv2.VideoCapture(0)
        
        print("--- EcoSort Live Feed Started ---")
        print("Press 'q' to quit the demo.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame temporarily for API
            temp_path = "live_frame.jpg"
            cv2.imwrite(temp_path, frame)

            # API Inference
            prediction_result = self.model.predict(temp_path, confidence=40).json()
            preds = prediction_result.get('predictions', [])

            if preds:
                for p in preds:
                    frame = self._draw_ui(frame, p)
                
                # Print current scan to terminal
                top_label = preds[0]['class'].upper()
                print(f"Detected: {top_label} ({preds[0]['confidence']:.1%})", end='\r')

            # Show the live feed
            cv2.imshow("EcoSort G16 - LIVE SCANNER", frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo Stopped.")

if __name__ == "__main__":
    MY_KEY = "DcJDGFd9CDdi4yOTljMJ"
    scanner = EcoSortLive(api_key=MY_KEY)
    scanner.run_live()