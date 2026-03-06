import boto3
import cv2

rekognition = boto3.client("rekognition", region_name="ap-south-1")

video_path = "test_video.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

print("\n--- SurakshaNet Hybrid Crowd Detection Started ---\n")

frame_skip = 30   # Process every 30th frame 
frame_count = 0
max_people_detected = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    
    if frame_count % frame_skip != 0:
        continue

    
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        continue

    image_bytes = buffer.tobytes()

    try:
        
        face_response = rekognition.detect_faces(
            Image={'Bytes': image_bytes}
        )
        face_count = len(face_response["FaceDetails"])

        
        label_response = rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,
            MinConfidence=80
        )

        person_count = 0
        for label in label_response["Labels"]:
            if label["Name"] == "Person":
                person_count = len(label["Instances"])

        
        final_count = max(face_count, person_count)
        if final_count > max_people_detected:
            max_people_detected = final_count

        print("\n==== VIDEO SUMMARY ====")
        print("Maximum People Detected in Video:", max_people_detected)

        if max_people_detected < 10:
              risk = "LOW"
        elif 10 <= max_people_detected < 30:
              risk = "MEDIUM"
        else:
            risk = "HIGH"

        print("Overall Risk Level:", risk)
        print("----------------------------------")

    except Exception as e:
        print("Error:", e)
    cv2.imshow("street CCTV camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n--- Video Processing Completed ---")