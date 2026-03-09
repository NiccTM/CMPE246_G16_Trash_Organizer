import cv2
import numpy as np
from ultralytics import YOLO
import os

class EcoSortSystem:
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize the AI model and sorting logic."""
        # Load the Nano model (best for Raspberry Pi performance)
        self.model = YOLO(model_path)
        
        # Mapping AI labels to physical waste bins
        self.bin_logic = {
            "RECYCLE": ["bottle", "cup", "wine glass", "can", "vase", "bowl"],
            "COMPOST": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot"],
            "LANDFILL": ["spoon", "fork", "knife", "cell phone", "backpack"]
        }
        
        # Translation layer for cleaner output
        self.friendly_names = {
            "vase": "Aluminum Can",
            "cup": "Disposable Cup/Can",
            "bottle": "Plastic/Glass Bottle"
        }

    def _apply_filters(self, img):
        """Preprocesses the image to help the AI find metallic edges."""
        # Pass 1: Original (No changes)
        # Pass 2: Sharpened
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(img, -1, kernel)
        
        # Pass 3: Contrast Enhanced (CLAHE) - Great for glare on cans
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        contrast_3ch = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
        
        return [img, sharp, contrast_3ch]

    def get_target_bin(self, label):
        """Determines the correct bin based on the AI label."""
        for bin_name, labels in self.bin_logic.items():
            if label in labels:
                return bin_name
        return "LANDFILL (Unknown)"

    def analyze_waste(self, image_path):
        """Main pipeline: Preprocess -> Multi-Pass Inference -> Map Result."""
        if not os.path.exists(image_path):
            return None, f"Error: {image_path} not found."

        raw_img = cv2.imread(image_path)
        variations = self._apply_filters(raw_img)
        
        print(f"--- Starting Analysis on {image_path} ---")
        
        for i, img_variant in enumerate(variations):
            # conf=0.05 is used to catch low-confidence items like the 'vase' can
            results = self.model(img_variant, conf=0.05, verbose=False)
            
            if len(results[0].boxes) > 0:
                # Get detection with highest confidence in this pass
                box = results[0].boxes[0]
                label = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                # Format for the user
                item_name = self.friendly_names.get(label, label.capitalize())
                bin_destination = self.get_target_bin(label)
                
                print(f"Success on Pass {i+1}!")
                
                result_data = {
                    "item": item_name,
                    "raw_label": label,
                    "confidence": f"{conf:.1%}",
                    "bin": bin_destination
                }
                return result_data, results[0].plot()

        return None, variations[1] # Return sharpened image for debugging if all fail

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize System
    ecosort = EcoSortSystem()
    
    # 2. Set file path for testing (ensure this image exists in the same directory or provide a valid path)
    test_image = "test_image.jpg" 
    
    # 3. Process
    data, visual_output = ecosort.analyze_waste(test_image)

    # 4. Display Results
    print("\n" + "="*30)
    if data:
        print(f"IDENTIFIED: {data['item']}")
        print(f"CONFIDENCE: {data['confidence']}")
        print(f"ACTION    : DISPOSE IN {data['bin']}")
        
        cv2.imshow("EcoSort - Detection Result", visual_output)
    else:
        print("RESULT: No objects detected after 3 passes.")
        print("TIP: Try a different angle or better lighting.")
        if visual_output is not None:
            cv2.imshow("EcoSort - Debug (Sharpened)", visual_output)
    
    print("="*30 + "\n")
    print("Press any key on the image window to close.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()