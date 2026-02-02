# SurakshaNet - System Design Document

## Overview
SurakshaNet is an AI-Powered crowd Disaster prevention and Response System designed for public gatherings in Bharat.

It uses real-time video analytics and predictive modeling to detect  overcrowding and prevent + response stampedes.

## System Architecture

Camera Feed (CCTV/Drone)
        |
        V
Amazon Rekognition (person detection)
        |
        V
Crowd Density + Tracking Module
        |
        V
Amazon SageMaker (Risk Prediction Model)
        |
        V
AWS Lambda (Trigger Emergency Action)
        |
        V
Amazon SNS ( SMS/App alerts)
        |
        V
Quicksight Dashboard for authorities

## Core Modules
1. Video Input Layer
- Live CCTV or drone stream ingestion
- Frames Extracted at regular intervals

2. Crowd Detection Module
- Uses YOLOv8 or AMazon Rekognition
- Detects individuals and counts crowd size
# Output:
- Total people count
- zone-wise density values

3. Tracking and Movement Analysis 
- DeepSort tracking assigns IDs to individuals
- Measures movement speed and direction
# Detects:
- Sudden rush
- Bidirectional crowd pressure
- Panic movement
4. Risk Prediction Engine 
- Implemented using Amazon sageMaker:
# inputs:
- Crowd density
- Movement anamalies
- Historical thresholds
# Outputs:
- Risk Score (0-100)
- Alert Level (Low/medium/high)

5. Alert and Response System
     AWS Lambda trigges actions when risk is high:
- Sends warning alerts through Amazon SNS
- NOtifies police/control rooms instantly

6. Monitoring Dashboard
     Amazon Quicksight dashboiard provides:
- Live crowd count 
- Heatmap risk zones 
- Alerts history
- Predictive safety insights

## Deployment Plan
- Edge device handle video feed capture
- AWS cloud processes analytics and prediction
- Scalable deployment using EC2 and Serverless tools

## Security and Privacy Considerations
- No facial identification stored
- Only crowd statistics used 
- Secure encrypted transmission of video metadata

## Drone-Assisted Emergency Response Layer 
    SurakshaNet can be extended beyond monitoring into active disaster response using an aerial drone system.

### Mother Drone + Mini Drone Swarm Concept
- A central Mother Drone provides high-altitude wide-area surveillance.
- Multiple lightweight Mini Drones can be deployed dynamically during emergencies.

### Key Benefits
- Rapid deployment in temporary high-density gatherings such as festivals, rallies, and pilgrimages  
- Eliminates the need for costly and time-consuming temporary CCTV installation  
- Provides flexible monitoring coverage across large open areas  

### Emergency Evacuation Support
During stampede escalation, mini drones can assist by:
- Broadcasting multilingual evacuation instructions  
- Providing audio/visual guidance toward safe exits  
- Helping authorities identify blocked routes or high-pressure zones  
- Improving crowd movement control in real time  

This drone-assisted evacuation guidance can significantly reduce casualties by enabling rapid crowd control and safe movement during emergencies.