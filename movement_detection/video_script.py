# import cv2
# from ultralytics import YOLO

# model = YOLO("best.pt")  # Trained to detect only "moving" objects

# cap = cv2.VideoCapture("movement.mp4")
# output_size = (800, 800)
# fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("output_only_moving.mp4", fourcc, fps, output_size)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, output_size)
#     results = model(frame)[0]
#     boxes = results.boxes

#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         label = "Moving"
#         color = (0, 255, 0)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("YOLOv8 - Moving Only", frame)
#     out.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()








# import cv2
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# # Load model and video
# model = YOLO("best.pt")
# cap = cv2.VideoCapture("movement.mp4")

# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = 0
# detection_frames = {}  # {track_id: frame_count}

# # Set output video size
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("output_with_boxes.mp4", fourcc, fps, (width, height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_count += 1

#     # Run detection with tracking enabled
#     results = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]

#     for box in results.boxes:
#         cls_id = int(box.cls[0])
#         if cls_id != 0:  # Assuming 0 is your target class (person or plate)
#             continue

#         # Use tracking ID
#         track_id = int(box.id[0]) if box.id is not None else -1
#         if track_id == -1:
#             continue

#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Count frames detected per track_id
#         detection_frames[track_id] = detection_frames.get(track_id, 0) + 1

#     out.write(frame)
#     cv2.imshow("Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # Convert frame counts to seconds
# detection_seconds = {k: v / fps for k, v in detection_frames.items()}

# # Plotting
# plt.bar(detection_seconds.keys(), detection_seconds.values(), color='skyblue')
# plt.xlabel("Object ID")
# plt.ylabel("Seconds Detected")
# plt.title("Detection Duration per Object")
# plt.grid(True)
# plt.tight_layout()
# plt.show()












import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load your trained YOLO model
model = YOLO("best.pt")

# Load the video
cap = cv2.VideoCapture("movement.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_with_ids.mp4", fourcc, fps, (width, height))

# Dictionary to count detection duration per ID
detection_frames = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and track using ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # Adjust based on your target class (e.g., person or plate)
            continue

        # Get the persistent ID assigned by ByteTrack
        if box.id is None:
            continue  # skip if no tracking ID
        track_id = int(box.id[0])

        # Count how many frames this ID appears
        detection_frames[track_id] = detection_frames.get(track_id, 0) + 1

        # Draw bounding box and ID
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Convert frames to seconds for each ID
detection_seconds = {k: v / fps for k, v in detection_frames.items()}

# Plot duration of appearance
plt.bar(detection_seconds.keys(), detection_seconds.values(), color='orange')
plt.xlabel("Object ID")
plt.ylabel("Seconds Detected")
plt.title("Time Detected per Object (Tracked)")
plt.grid(True)
plt.tight_layout()
plt.show()
