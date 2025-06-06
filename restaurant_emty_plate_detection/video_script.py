import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Load video (.mov format)
video_path = "v3.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 560))

    # Use stream mode for better performance
    results = model.predict(source=frame, stream=True, conf=0.25)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])  # confidence score

            if class_name == "empty_plate":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name} ({conf * 100:.1f})"
                
                # Draw bounding box and label with confidence
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Empty Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
