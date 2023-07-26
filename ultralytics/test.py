from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2

model = YOLO(r'C:\Users\Alpha 15 A3DD\Z-coding\Fire-Detection-using-YOLOv8\best.pt')

res = model.predict(r'C:\Users\Alpha 15 A3DD\Z-coding\Fire-Detection-using-YOLOv8\fire.jpg',show=True)
# print(res)