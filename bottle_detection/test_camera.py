# from ultralytics import YOLO
# import cv2

# model = YOLO("bottle_detection1.pt")
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




# # timer for continues 1 min detection
# from ultralytics import YOLO
# import cv2
# import time

# model = YOLO("bottle_detection1.pt")
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Cannot open webcam")
#     exit()

# # Variables to track detection time
# detected_start_time = None
# ALERT_THRESHOLD_SECONDS = 5 
# alert_triggered = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Can't receive frame")
#         break

#     results = model(frame, conf=0.1)
#     boxes = results[0].boxes

#     # Check if bottle is detected
#     if boxes is not None and len(boxes) > 0:
#         if detected_start_time is None:
#             detected_start_time = time.time()

#         elapsed_time = time.time() - detected_start_time

#         if elapsed_time >= ALERT_THRESHOLD_SECONDS:
#             alert_triggered = True
#     else:
#         detected_start_time = None
#         alert_triggered = False

#     # Draw detections
#     annotated_frame = results[0].plot()

#     # Show alert message on frame if needed
#     if alert_triggered:
#         cv2.putText(
#             annotated_frame,
#             "Bottle detected for 5 second!",
#             (30, 50),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),  # Red color
#             3
#         )

#     # Display the result
#     cv2.imshow("YOLOv8 Live Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()










# yolo modelfrom ultralytics import YOLOfrom ultralytics import YOLOfrom ultralytics 
from ultralytics import YOLO
import cv2

# Load YOLOv8n model (pre-trained on COCO)
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame")
        break

    # Inference
    results = model(frame, conf=0.4)[0]  # single image

    # Extract class names from model
    class_names = model.names

    bottle_boxes = []
    table_boxes = []

    # Loop through all detected boxes
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = class_names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box

        if class_name == "bottle":
            bottle_boxes.append((x1, y1, x2, y2))
        elif "table" in class_name or "dining" in class_name:
            table_boxes.append((x1, y1, x2, y2))

    annotated_frame = frame.copy()

    # Check each bottle with each table
    for (bx1, by1, bx2, by2) in bottle_boxes:
        bottle_center_x = (bx1 + bx2) // 2
        bottle_bottom_y = by2

        for (tx1, ty1, tx2, ty2) in table_boxes:
            # Check if bottle's bottom is inside table bounds
            if tx1 < bottle_center_x < tx2 and ty1 < bottle_bottom_y < ty2:
                # Draw the box and label
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    "Bottle on Table",
                    (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                break  # Skip other tables once matched

    # Show result
    cv2.imshow("Bottle on Table Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()








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















# from ultralytics import YOLO
# import cv2
# import time

# # Load model
# model = YOLO("best.pt")

# # Start webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Cannot open webcam")
#     exit()

# last_time = 0
# fps = 1  # Adjust FPS limit

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Can't receive frame")
#         break

#     current_time = time.time()
#     if current_time - last_time >= 1 / fps:
#         last_time = current_time

#         # Run inference
#         results = model(frame, conf=0.1)
#         result = results[0]

#         # Copy frame for annotation
#         annotated_frame = frame.copy()

#         # Draw only boxes with confidence >= 0.2
#         for box in result.boxes:
#             conf = float(box.conf[0])
#             if conf >= 0.2:
#                 cls_id = int(box.cls[0])
#                 class_name = model.names[cls_id]
#                 xyxy = box.xyxy[0].cpu().numpy().astype(int)
#                 label = f"{class_name} {conf:.2f}"
                
#                 # Draw box
#                 cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
#                 cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#         # Show result
#         cv2.imshow("YOLOv8 - Conf >= 0.4", annotated_frame)

#     # Quit key
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# //gray scale


# from ultralytics import YOLO
# import cv2

# model = YOLO("glass.pt")
# # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://forthtech-food:tyttet-cejvam-hiSmu0@192.168.1.41:554/stream1")
# if not cap.isOpened():
#     print("Error: Cannot open webcam")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Can't receive frame")
#         break

#     # Convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Convert grayscale to 3-channel BGR for YOLO input
#     gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

#     # Run YOLO detection
#     results = model(gray_bgr, conf=0.1)
#     print(results[0].boxes)

#     # Annotated result
#     annotated_frame = results[0].plot()

#     # Show in grayscale
#     display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("YOLOv8 Live Detection (Grayscale)", display_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
