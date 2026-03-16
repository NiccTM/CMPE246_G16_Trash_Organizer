import cv2
import numpy as np
import os
from roboflow import Roboflow

class EcoSortSystem:
    def __init__(self, api_key):
        """Initialize using standard Roboflow library."""
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("garbage-classification-3")
        self.model = project.version(2).model
        
        # Mapping for the 'Garbage Classification 3' dataset
        self.bin_logic = {
            "RECYCLE": ["cardboard", "glass", "metal", "paper", "plastic"],
            "COMPOST": ["biodegradable"],
            "LANDFILL": ["cloth", "trash"]
        }

    def _draw_ui(self, img, pred):
        """Draws bounding boxes and labels."""
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        # Draw styling
        color = (0, 255, 0) # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_str = f"{pred['class'].upper()} {pred['confidence']:.1%}"
        cv2.putText(img, label_str, (x1, max(20, y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    def analyze_waste(self, image_path):
        """Inference pipeline with local file saving."""
        if not os.path.exists(image_path):
            return None, None

        print(f"--- Analyzing {image_path} via Roboflow Cloud ---")
        prediction_result = self.model.predict(image_path, confidence=40).json()
        
        display_img = cv2.imread(image_path)
        preds = prediction_result.get('predictions', [])

        if preds:
            top_pred = preds[0]
            label = top_pred['class'].lower()
            
            target_bin = "LANDFILL"
            for b_name, materials in self.bin_logic.items():
                if label in materials:
                    target_bin = b_name
                    break
            
            annotated_img = self._draw_ui(display_img, top_pred)
            
            # --- Engineering Log: Save the result ---
            output_name = f"result_{label}.jpg"
            cv2.imwrite(output_name, annotated_img)
            print(f"DEBUG: Saved annotated image as {output_name}")
            
            return {
                "material": label.upper(),
                "confidence": f"{top_pred['confidence']:.1%}",
                "bin": target_bin
            }, annotated_img

        return None, display_img

# --- Main Execution ---
if __name__ == "__main__":
    MY_KEY = "DcJDGFd9CDdi4yOTljMJ" 
    ecosort = EcoSortSystem(api_key=MY_KEY)
    
    test_image = "test_image.jpg"
    data, visual = ecosort.analyze_waste(test_image)

    print("\n" + "="*30)
    if data:
        print(f"IDENTIFIED: {data['material']}")
        print(f"CONFIDENCE: {data['confidence']}")
        print(f"ACTION    : DISPOSE IN {data['bin']}")
        
        # Ensure we have a valid window-capable OpenCV version
        try:
            # Resize for display if the image is massive
            visual = cv2.resize(visual, (800, 600)) 
            cv2.imshow("EcoSort G16 - Roboflow GC3", visual)
            print("\nDisplaying window... Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("\n[ERROR] Could not open display window.")
            print("Ensure you ran: pip install opencv-python")
    else:
        print("No items detected.")
    print("="*30 + "\n")