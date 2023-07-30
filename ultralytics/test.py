from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2

model = YOLO('/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/best.pt')

# res = model.predict('/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/fire.jpg',show=True)
res = model.predict('/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/ultralytics/yolo/v8/detect/demo.mp4',show=True)
# print(res)