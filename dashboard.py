import streamlit as st
import cv2
import boto3
import numpy as np
from ultralytics import YOLO
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🚨 SurakshaNet Crowd Monitoring Dashboard")

rekognition = boto3.client("rekognition", region_name="ap-south-1")
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
table = dynamodb.Table('SurakshaNetAnalytics')

sns = boto3.client('sns', region_name='ap-south-1')
SNS_TOPIC_ARN = "arn:aws:sns:ap-south-1:593737078358:surakshaNetAlerts"

model = YOLO("yolov8n.pt")

video1 = cv2.VideoCapture("test_video.mp4")
video2 = cv2.VideoCapture("test_video1.mp4")
video3 = cv2.VideoCapture("short.mp4")

entry_count = 0
exit_count = 0
previous_positions = {}

prev_gray = None
grid_size = 3

col1, col2 = st.columns([3,1])
col3, col4 = st.columns([3,1])
col5, col6 = st.columns([3,1])

video1_frame = col1.empty()
data1 = col2.empty()

video2_frame = col3.empty()
data2 = col4.empty()

video3_frame = col5.empty()
data3 = col6.empty()

while True:

    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    ret3, frame3 = video3.read()

    if not ret1 and not ret2 and not ret3:
        break

    if ret1:

        success, buffer = cv2.imencode(".jpg", frame1)
        image_bytes = buffer.tobytes()

        face_response = rekognition.detect_faces(Image={'Bytes': image_bytes})
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

        if final_count < 10:
            risk = "LOW"
        elif final_count < 30:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        video1_frame.image(frame1, channels="BGR", use_container_width=True)

        data1.markdown(f"""
        ### Camera 1 Analytics
         People Count: **{final_count}**

        ⚠ Risk Level: **{risk}**
        """)

    if ret2:

        results = model.track(frame2, persist=True, classes=[0])

        frame_height = frame2.shape[0]
        line_position = frame_height // 2

        for result in results:

            boxes = result.boxes
            if boxes.id is None:
                continue

            for box, track_id in zip(boxes.xyxy, boxes.id):

                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)

                center_y = (y1 + y2) // 2

                cv2.rectangle(frame2,(x1,y1),(x2,y2),(0,255,0),2)

                if track_id in previous_positions:
                    prev_y = previous_positions[track_id]

                    if prev_y < line_position and center_y >= line_position:
                        entry_count += 1

                    if prev_y > line_position and center_y <= line_position:
                        exit_count += 1

                previous_positions[track_id] = center_y

        cv2.line(frame2,(0,line_position),(frame2.shape[1],line_position),(255,0,0),2)

        video2_frame.image(frame2, channels="BGR", use_container_width=True)

        data2.markdown(f"""
        ### Camera 2 Analytics
         Entry Count: **{entry_count}**

        Exit Count: **{exit_count}**
        """)

    if ret3:

        height, width, _ = frame3.shape
        gray = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray,100,200)

        cell_h = height//grid_size
        cell_w = width//grid_size

        max_local_density = 0

        for i in range(grid_size):
            for j in range(grid_size):

                y1=i*cell_h
                y2=(i+1)*cell_h
                x1=j*cell_w
                x2=(j+1)*cell_w

                cell = edges[y1:y2,x1:x2]
                density = np.sum(cell>0)

                if density > max_local_density:
                    max_local_density = density

        density_score = max_local_density/(cell_h*cell_w)

        flow_score = 0


        if prev_gray is not None:
            flow=cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
            magnitude,_=cv2.cartToPolar(flow[...,0],flow[...,1])
            flow_score=np.mean(magnitude)

        prev_gray = gray

        risk_score = (density_score*0.6)+(flow_score*0.4)

        if risk_score < 1:
            risk_level="LOW"
        elif risk_score < 3:
            risk_level="MEDIUM"
        else:
            risk_level="HIGH"

        timestamp=datetime.utcnow().isoformat()

        table.put_item(
            Item={
                'timestamp':timestamp,
                'risk_score':str(risk_score),
                'risk_level':risk_level
            }
        )

        if risk_level=="HIGH":
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=f"Extreme Crowd Risk Detected {risk_score}",
                Subject="SurakshaNet Alert"
            )

        video3_frame.image(frame3, channels="BGR", use_container_width=True)

        data3.markdown(f"""
        ### Camera 3 Analytics

         Density Score: **{density_score:.3f}**

         Flow Score: **{flow_score:.3f}**

         Risk Level: **{risk_level}**
        """)

video1.release()
video2.release()
video3.release()