import cv2
from ultralytics import YOLO

# Load pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Use webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Object Detection and Tracking started.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from webcam.")
        break

    # Detect and track objects
    results = model.track(frame, persist=True, verbose=False)

    # Draw bounding boxes and tracking IDs
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if box.id is not None:
                track_id = int(box.id[0])
                label = f"ID {track_id} - {class_name} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Object Detection and Tracking", frame)

    # Press q to close window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()