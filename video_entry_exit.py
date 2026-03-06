from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt")

video_path = "test_video1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()


entry_count = 0
exit_count = 0


previous_positions = {}

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


line_position = frame_height // 2

print("\n--- Entry/Exit Detection Started ---\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])  

    for result in results:
        boxes = result.boxes

        if boxes.id is None:
            continue

        for box, track_id in zip(boxes.xyxy, boxes.id):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0,0,255), -1)

            if track_id in previous_positions:
                prev_y = previous_positions[track_id]

                
                if prev_y < line_position and center_y >= line_position:
                    entry_count += 1

                
                if prev_y > line_position and center_y <= line_position:
                    exit_count += 1

            previous_positions[track_id] = center_y

    cv2.line(frame, (0, line_position), (frame_width, line_position), (255,0,0), 2)
    cv2.putText(frame, f"Entry: {entry_count}", (20,40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Exit: {exit_count}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Entry/Exit Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n===== FINAL COUNT =====")
print("Total Entry:", entry_count)
print("Total Exit:", exit_count)