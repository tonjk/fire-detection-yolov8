from ultralytics import YOLO
# from ultralytics.yolo.engine.model import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import numpy as np

import cv2

image = True

if image:
    print('hello')
    model = YOLO('/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/best.pt')
    img_path = '/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/fire.jpg'
    # img_path = '/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/ffire.jpeg'
    # img_path = '/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/ultralytics/yolo/v8/detect/demo.mp4'
    # img = cv2.imread(img_path)
    # res = model(img)
    res = model.predict(img_path, show=False)
    print(res)
    print(type(res))
    print(model.model.names)
    print('---------------------------')
    # print(res[-1])
    # print(res[-1].tolist())
    img = cv2.imread(img_path)
    draw_img = img.copy()

    for *xywh, conf, lb in res[0].boxes.data.tolist():
        frame_color = (0,0,255)
        cv2.rectangle(draw_img, tuple(map(int,(xywh[0], xywh[1]))), tuple(map(int,(xywh[2], xywh[3]))), (0,0,255), thickness=1)
        name_label = f"{model.model.names[int(lb)]}_{np.round(conf*100, 2)}"
        (label_width, label_height), baseline = cv2.getTextSize(name_label , cv2.FONT_HERSHEY_PLAIN,1,1)
        top_left = tuple(map(int,[int(xywh[0]),int(xywh[1])-(label_height+baseline+5)]))
        top_right = tuple(map(int,[int(xywh[0])+label_width,int(xywh[1])]))
        org = tuple(map(int,[int(xywh[0]),int(xywh[1])-baseline]))
        cv2.rectangle(draw_img, (int(xywh[0]), int(xywh[1])), (int(xywh[2]), int(xywh[3])), frame_color, 1)
        cv2.rectangle(draw_img, top_left, top_right, frame_color, -1)
        cv2.putText(draw_img, name_label, org, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    cv2.imshow('result', draw_img)
    cv2.waitKey(0) & 0xFF == ord("q")
    cv2.destroyAllWindows()

else:
    model = YOLO('/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/best.pt')
    # model = YOLO('./new_model/yolov8s.pt')
    # cap = cv2.VideoCapture('/Users/tonjk/Desktop/BDA/EGAT/model-fire/fire-detection-yolov8/ultralytics/yolo/v8/detect/demo.mp4')
    cap = cv2.VideoCapture(0)
    # res = model(source=0, stream=True, show=True)
    print('Frame size :', cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 1
    while True:
        ret, frame = cap.read()
        assert ret
        # frame = cv2.resize(frame, (640,460))
        res = model.predict(frame, conf=0.25, imgsz=640, show=False, verbose=False)
        # print(frame.shape)
        draw_img = frame.copy()
        for *xywh, conf, lb in res[0].boxes.data.tolist():
            frame_color = (0,0,255)
            cv2.rectangle(draw_img, tuple(map(int,(xywh[0], xywh[1]))), tuple(map(int,(xywh[2], xywh[3]))), (0,0,255), thickness=1)
            name_label = f"{model.model.names[int(lb)]}_{np.round(conf*100, 2)}"
            (label_width, label_height), baseline = cv2.getTextSize(name_label , cv2.FONT_HERSHEY_PLAIN,1,1)
            top_left = tuple(map(int,[int(xywh[0]),int(xywh[1])-(label_height+baseline+5)]))
            top_right = tuple(map(int,[int(xywh[0])+label_width,int(xywh[1])]))
            org = tuple(map(int,[int(xywh[0]),int(xywh[1])-baseline]))
            cv2.rectangle(draw_img, (int(xywh[0]), int(xywh[1])), (int(xywh[2]), int(xywh[3])), frame_color, 1)
            cv2.rectangle(draw_img, top_left, top_right, frame_color, -1)
            cv2.putText(draw_img, name_label, org, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)


        cv2.putText(draw_img, f"{count}", (frame.shape[1]-100, frame.shape[0]-20),cv2.FONT_HERSHEY_PLAIN, 3, 1, 3)            
        count+=1    
        cv2.imshow('result', draw_img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()