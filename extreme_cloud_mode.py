import cv2
import numpy as np
import boto3
from datetime import datetime

dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
table = dynamodb.Table('SurakshaNetAnalytics')

sns = boto3.client('sns', region_name='ap-south-1')
SNS_TOPIC_ARN = "arn:aws:sns:ap-south-1:593737078358:surakshaNetAlerts"

video_path = "short.mp4"
cap = cv2.VideoCapture(video_path)

prev_gray = None
grid_size = 3

print("\n--- EXTREME CLOUD MODE STARTED ---\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    edges = cv2.Canny(gray, 100, 200)
    cell_h = height // grid_size
    cell_w = width // grid_size

    max_local_density = 0

    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            cell = edges[y1:y2, x1:x2]
            density = np.sum(cell > 0)

            if density > max_local_density:
                max_local_density = density

    density_score = max_local_density / (cell_h * cell_w)

    
    flow_score = 0
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_score = np.mean(magnitude)

    prev_gray = gray

    
    risk_score = (density_score * 0.6) + (flow_score * 0.4)

    if risk_score < 1:
        risk_level = "LOW"
    elif risk_score < 3:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    print("Risk Level:", risk_level)

    
    timestamp = datetime.utcnow().isoformat()

    table.put_item(
        Item={
            'timestamp': timestamp,
            'risk_score': str(risk_score),
            'risk_level': risk_level
        }
    )

    
    if risk_level == "HIGH":
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=f"⚠ Extreme Crowd Risk Detected!\nRisk Score: {risk_score}",
            Subject="SurakshaNet ALERT"
        )
    cv2.imshow("SurakshaNet Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print("\n--- Cloud Processing Completed ---")