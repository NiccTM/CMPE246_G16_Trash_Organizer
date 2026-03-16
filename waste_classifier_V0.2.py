import cv2
import numpy as np
import os
from roboflow import Roboflow
from collections import Counter

class EcoSortSystem:
    def __init__(self, api_key):
        """Initialize Roboflow and Model Logic."""
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("garbage-classification-3")
        self.model = project.version(2).model
        
        # Sorting logic mapping
        self.bin_logic = {
            "RECYCLE": ["cardboard", "glass", "metal", "paper", "plastic"],
            "COMPOST": ["biodegradable"],
            "LANDFILL": ["cloth", "trash"]
        }

    def _draw_ui(self, img, pred):
        """Standard Bounding Box UI with pixel coordinates."""
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        
        color = (0, 255, 0) # Green for detection
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_str = f"{pred['class'].upper()} {pred['confidence']:.0%}"
        cv2.putText(img, label_str, (x1, max(20, y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def analyze_waste(self, image_path):
        """Processes objects with NMS filtering to prevent double-counting."""
        if not os.path.exists(image_path):
            return None, "File Not Found"

        print(f"--- Processing {image_path} with NMS filtering ---")
        prediction_result = self.model.predict(image_path, confidence=35).json()
        
        display_img = cv2.imread(image_path)
        preds = prediction_result.get('predictions', [])

        # Prepare data for OpenCV's NMS function
        boxes = []
        confidences = []
        
        for p in preds:
            w, h = int(p['width']), int(p['height'])
            x = int(p['x'] - w/2)
            y = int(p['y'] - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(p['confidence']))

        # Apply NMS: removes boxes that overlap by more than 40% (nms_threshold=0.4)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.35, nms_threshold=0.4)

        detected_items = []
        if len(indices) > 0:
            # indices can be a list or a nested numpy array depending on OpenCV version
            for i in indices.flatten():
                p = preds[i]
                label = p['class'].lower()
                
                target_bin = "LANDFILL"
                for b_name, materials in self.bin_logic.items():
                    if label in materials:
                        target_bin = b_name
                        break
                
                detected_items.append({
                    "material": label.upper(),
                    "bin": target_bin,
                    "confidence": p['confidence']
                })
                
                display_img = self._draw_ui(display_img, p)
            
            return detected_items, display_img

        return [], display_img

# --- Main Execution ---
if __name__ == "__main__":
    MY_KEY = "DcJDGFd9CDdi4yOTljMJ" 
    ecosort = EcoSortSystem(api_key=MY_KEY)
    
    test_image = "test_image.jpg"
    results, visual = ecosort.analyze_waste(test_image)

    print("\n" + "="*45)
    if results:
        print(f"SUCCESS: {len(results)} UNIQUE ITEMS IDENTIFIED")
        
        # Tally the results
        tally = Counter([item['material'] for item in results])
        for material, count in tally.items():
            print(f" - {material}: {count}")
        
        print("-" * 45)
        for i, item in enumerate(results, 1):
            print(f"[{i}] {item['material']} -> {item['bin']} ({item['confidence']:.1%})")
        
        cv2.imshow("EcoSort G16 - Clean Multi-Detection", visual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("RESULT: No items detected.")
    print("="*45 + "\n")