# from ultralytics import YOLO
# import cv2

# model = YOLO("best.pt")
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Cannot open webcam")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Can't receive frame")
#         break

#     results = model(frame, conf=0.1)
#     print(results[0].boxes) 

#     annotated_frame = results[0].plot()

#     cv2.imshow("YOLOv8 Live Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



#less frames per second


# from ultralytics import YOLO
# import cv2
# import time

# model = YOLO("best.pt")
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Cannot open webcam")
#     exit()

# last_time = 0
# fps = 1  

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Can't receive frame")
#         break

#     current_time = time.time()
    
#     # Process only if 1/fps seconds have passed
#     if current_time - last_time >= 1 / fps:
#         last_time = current_time

#         results = model(frame, conf=0.1)
#         print(results[0].boxes)

#         annotated_frame = results[0].plot()
#         cv2.imshow("YOLOv8 - 5 FPS", annotated_frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("best.pt")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

last_time = 0
fps = 1  # Adjust FPS limit

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    current_time = time.time()
    if current_time - last_time >= 1 / fps:
        last_time = current_time

        # Run inference
        results = model(frame, conf=0.1)
        result = results[0]

        # Copy frame for annotation
        annotated_frame = frame.copy()

        # Draw only boxes with confidence >= 0.2
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf >= 0.2:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label = f"{class_name} {conf:.2f}"
                
                # Draw box
                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
                cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show result
        cv2.imshow("YOLOv8 - Conf >= 0.4", annotated_frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
