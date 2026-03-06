# SurakshaNet – AI Crowd Intelligence & Safety Monitoring System
## Overview
SurakshaNet is an AI-powered crowd monitoring and early risk detection system designed for large public spaces such as metro stations, pilgrimage sites, festivals, stadiums, and transportation hubs.
The system employs two complementary approaches for data analysis and AI-based decision making. The first approach focuses on people counting at entry and exit points, where the system monitors and analyzes the number of individuals entering and leaving a location to understand crowd flow and occupancy levels. The second approach is a density-based analysis designed for large gatherings, where CCTV footage is processed in real time to estimate crowd density, identify congestion zones, and generate early warnings for potential stampede risks. Together, these methods enable more accurate monitoring, improved situational awareness, and proactive crowd management.
By combining computer vision, AI-based density estimation, and cloud-based alert systems, SurakshaNet helps authorities monitor crowd situations and take preventive action before dangerous situations occur.
## Problem Statement
Large gatherings often lead to overcrowding and stampede risks due to lack of real-time crowd intelligence.
Traditional monitoring methods rely on:
1. manual CCTV observation
2. delayed response
3. limited situational awareness
This makes it difficult for authorities to detect dangerous crowd buildup early.
## Solution
SurakshaNet introduces an AI-powered crowd analytics platform that:
1. Processes CCTV video streams.
2. Generates crowd density heatmaps.
3. Divides surveillance areas into risk analysis grids.
4. Detects high-density zones in real time.
5. Sends automated alerts to control rooms.
This allows authorities to take proactive safety measures.
## Key Features
1. Real-Time Crowd Density Estimation:- AI analyzes video frames to estimate crowd density using computer vision techniques and segmentation models.
2. Dynamic Heatmap Visualization:- The system generates a live crowd heatmap, highlighting areas with higher crowd concentration.
3. Grid-Based Risk Detection:- The surveillance area is divided into multiple zones, and each grid calculates a crowd density score.
4. Color-based risk classification:
 Color                    Crowd Level
Green                     Low density
Yellow                    Medium density
Red                       High density
5. Automated Alerts :- When density exceeds a threshold, the system can trigger alerts to authorities through notification systems.
6. Control Room Dashboard:- Authorities can monitor crowd activity through a centralized dashboard displaying live analytics.
## System Architecture

CCTV Camera Feed
        ↓
Frame Processing
        ↓
AI Crowd Segmentation
        ↓
Density Heatmap Generation
        ↓
Grid-Based Risk Analysis
        ↓
Alert System
        ↓
Control Room Dashboard

## Technologies Used
### Programming:- 
Python
### Computer Vision:-
OpenCV
### AI Models:-
YOLOv8 Segmentation (Ultralytics), amazon rekongition
   
### Cloud Integration 
AWS Lambda
Amazon SNS
Amazon DynamoDB
Amazon QuickSight

## Example Output
The system produces:
AI-generated crowd heatmap
Density values for each surveillance grid
Risk visualization for crowd congestion zones
This enables real-time crowd monitoring and safety analysis.
## Drone Integration in SurakshaNet
Role of Drones
Drones act as mobile surveillance units that can monitor areas where fixed CCTV cameras cannot cover.
They provide:
1. aerial crowd monitoring
2. dynamic crowd density analysis
3. emergency response support

## Why Drones Are Important
### CCTV limitations:
1. fixed angle
2. blind spots
3. limited coverage
### Drones provide:
Advantage                      Description
Wide coverage                  Monitor large crowd areas
Dynamic positioning            Move to crowded zones
Aerial perspective             Better density estimation
Rapid response                 Reach emergency areas quickly
### Specially designed drones are integrated into the SurakshaNet system to enhance emergency response. In critical situations, the drones support crowd management by monitoring the area, issuing safety announcements, and helping guide people toward designated exit paths for safe evacuation.
## Future Improvements
1. Real-time CCTV video streaming
2. Crowd flow direction analysis
3. Stampede risk prediction using ML
4. Drone-based monitoring integration
5. Mobile alerts for authorities
6. Smart city integration
7. Drone integration (with special design to protect people)
## Use Cases
SurakshaNet can be deployed in:
1. Metro stations
2. Railway platforms
3. Religious gatherings (Kumbh Mela, pilgrimages)
4. Stadiums and concerts
5. Airports
6. Large public events
## Project Goal
The goal of SurakshaNet is to transform passive surveillance into intelligent crowd safety monitoring, enabling authorities to detect and respond to dangerous crowd situations before they escalate.
# Author
Team Surakshanet

