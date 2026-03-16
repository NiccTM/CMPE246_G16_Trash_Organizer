import cv2
from inference_sdk import InferenceHTTPClient

# 1. INITIALIZE THE BRAIN
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="DcJDGFd9CDdi4yOTljMJ" # Make sure this is your real key!
)

# 2. RUN THE SORTER
img_path = input("Enter image filename (e.g. test_image.jpg): ")
image = cv2.imread(img_path)

if image is None:
    print("Error: Could not open image. Check the filename!")
else:
    try:
        # Get predictions from Roboflow
        result = CLIENT.infer(img_path, model_id="garbage-classification-3/2")
        predictions = result.get('predictions', [])

        if predictions:
            for pred in predictions:
                # Extract coordinates for the box
                x = int(pred['x'])
                y = int(pred['y'])
                w = int(pred['width'])
                h = int(pred['height'])
                label = pred['class']
                conf = pred['confidence']

                # Calculate box corners for OpenCV
                # Roboflow gives center (x,y), OpenCV needs top-left/bottom-right
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)

                # Draw the rectangle (Green box, thickness 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add the label text above the box
                text = f"{label} {conf*100:.1f}%"
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"Detected: {label.upper()} ({conf*100:.1f}%)")

            # 3. SHOW THE PICTURE
            cv2.imshow("Trash Sorter AI", image)
            print("Image window opened! Press any key on your keyboard to close it.")
            cv2.waitKey(0) # This keeps the window open until you press a key
            cv2.destroyAllWindows()
            
        else:
            print("No objects detected.")

    except Exception as e:
        print(f"Error during classification: {e}")