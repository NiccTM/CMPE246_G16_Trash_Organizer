import cv2
import numpy as np
import os
from roboflow import Roboflow

class EcoSortScanner:
    def __init__(self, api_key):
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("garbage-classification-3")
        self.model = project.version(2).model
        
        self.bin_logic = {
            "RECYCLE": ["cardboard", "glass", "metal", "paper", "plastic"],
            "COMPOST": ["biodegradable"],
            "LANDFILL": ["cloth", "trash"]
        }

    def _get_bin(self, label):
        """Helper to find the correct bin."""
        for b_name, materials in self.bin_logic.items():
            if label.lower() in materials:
                return b_name
        return "LANDFILL"

    def _draw_ui(self, img, pred):
        """Draws bounding boxes and individual item labels."""
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Determine the bin for this specific item
        target_bin = self._get_bin(pred['class'])
        
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add the class AND the bin to the bounding box label
        label_str = f"{pred['class'].upper()} -> {target_bin}"
        cv2.putText(img, label_str, (x1, max(20, y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return img, target_bin

    def run_scanner(self):
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*40)
        print("EcoSort G16 Point-and-Shoot Scanner")
        print(" -> Press 'SPACEBAR' to capture and scan.")
        print(" -> Press 'Q' to quit.")
        print("="*40 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create a copy for the live preview
            preview = frame.copy()
            cv2.putText(preview, "Press SPACE to Scan | Q to Quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("EcoSort G16 - Viewfinder", preview)

            key = cv2.waitKey(1) & 0xFF
            
            # Press 'Q' to Quit
            if key == ord('q'):
                break
                
            # Press 'SPACEBAR' to Snap and Scan
            elif key == 32: 
                print("\nSnapping photo... sending to Cloud API...")
                
                # Show a temporary "Analyzing" screen
                cv2.putText(frame, "ANALYZING...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow("EcoSort G16 - Viewfinder", frame)
                cv2.waitKey(1) # Force UI to update
                
                # Save the clean frame (without the 'Analyzing' text) for the API
                temp_path = "snap.jpg"
                cv2.imwrite(temp_path, preview) 

                # Run Inference
                prediction_result = self.model.predict(temp_path, confidence=35).json()
                preds = prediction_result.get('predictions', [])

                # Reload the clean image to draw our final UI on
                result_img = cv2.imread(temp_path)

                if preds:
                    primary_bin = "LANDFILL"
                    
                    # Draw boxes for all items
                    for idx, p in enumerate(preds):
                        result_img, b_name = self._draw_ui(result_img, p)
                        # We use the highest confidence item (the first one) for the main banner
                        if idx == 0:
                            primary_bin = b_name

                    # --- UI Upgrade: Giant Status Banner ---
                    h, w, _ = result_img.shape
                    # Draw a black rectangle at the bottom
                    cv2.rectangle(result_img, (0, h-70), (w, h), (0, 0, 0), -1)
                    
                    # Add the primary sorting instruction
                    banner_text = f"ACTION: PLACE IN {primary_bin}"
                    cv2.putText(result_img, banner_text, (20, h-25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                    # Show the final result
                    print(f"Result: {banner_text}")
                    cv2.imshow("EcoSort G16 - Scan Result", result_img)
                    
                    print("Press ANY KEY on the image window to take another picture.")
                    cv2.waitKey(0) # Pauses the loop until user presses a key
                    cv2.destroyWindow("EcoSort G16 - Scan Result")
                else:
                    print("No items detected. Try adjusting the lighting or angle.")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Scanner shut down.")

if __name__ == "__main__":
    MY_KEY = "DcJDGFd9CDdi4yOTljMJ"
    scanner = EcoSortScanner(api_key=MY_KEY)
    scanner.run_scanner()