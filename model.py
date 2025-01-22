from ultralytics import YOLO
import os

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# path = r'F:\PROJECTS\criminalDetection\pictures\car.jpg'
# print(os.listdir(path))

# # Run inference on an image or video
results = model(r'F:\PROJECTS\criminalDetection\pictures\car.jpg')

for result in results:
    result.show() 

