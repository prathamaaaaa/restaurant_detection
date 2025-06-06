from ultralytics import YOLO
import cv2
import time

# Load models
object_model = YOLO("yolov8n.pt")   # COCO
glasses_model = YOLO("best.pt")     # Custom-trained glasses/goggles
keys_model = YOLO("keys.pt")        # Custom-trained keys

# Classes of interest
object_classes = ['person', 'bottle', 'chair', 'spoon']
glasses_classes = ['glass', 'goggles']  # Make sure these match your training labels
keys_classes = ['Cabinet key', 'Room key', 'Vehicle key']         # Adjust based on your dataset classes

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("🚀 Running multi-model detection... Press 'q' to quit.")
last_time = 0
fps = 5  # Inference speed control

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    current_time = time.time()
    if current_time - last_time < 1 / fps:
        continue
    last_time = current_time

    # Run inference for all 3 models
    object_results = object_model(frame, conf=0.3)
    glasses_results = glasses_model(frame, conf=0.1)
    keys_results = keys_model(frame, conf=0.2)

    annotated_frame = frame.copy()

    # --- Object Detection ---
    for result in object_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = object_model.names[cls_id]
            if class_name in object_classes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- Glasses Detection ---
    for result in glasses_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = glasses_model.names[cls_id]
            if class_name.lower() in glasses_classes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Keys Detection ---
    for result in keys_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = keys_model.names[cls_id]
            if class_name.lower() in keys_classes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
                cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show final annotated frame
    cv2.imshow("🎯 Multi-Model Detection (Objects + Glasses + Keys)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
