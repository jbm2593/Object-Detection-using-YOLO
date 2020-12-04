## 라즈베리파이 mjpeg로부터 서버로 영상 넘겨서 opencv로 확인 코드
import cv2 as cv
import numpy as np
from PIL import Image
from pytesseract import *
import os
from gtts import gTTS


cap = cv.VideoCapture("http://220.79.109.141:8090/?action=stream")

# while (True):
#
#     ret, img_color = cap.read()
#
#     if ret == False:
#         continue;
#
#     cv.imshow('bgr', img_color)
#
#     # ESC 키누르면 종료
#     if cv.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv.destroyAllWindows()
# 라즈베리파이 mjpeg로부터 서버로 영상 넘겨서 opencv로 확인 코드 여기까지!


# Yolo 로드
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 이미지 가져오기
# 다음 물체 감지를 할 이미지를 로드하고 너비, 높이도 가져온다
# img = cv.imread("sample3.jpg")


while (True):
    ret, img_color = cap.read()

    img = cv.resize(img_color, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    # text_count = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label, (x, y + 30), font, 3, color, 3)
            # print(label) ##물체 이름
            # text_count.append(label)

    # cv.namedWindow('streaming', cv.WIDNOW_AUTOSIZE)
    cv.imshow("streaming", img)
    # print(text_count) ##리스트변수에 어떻게 담기는지 출력
    # print(label) ##함수 끝나고 출력되면 맨 뒤에것만 출력된다 원인은 아직 모르겠음

    # # mp3파일 저장하고 출려하는데 소리 출력이 안됨(데스크탑만 안됨)
    # mytext = 'Hello Python'
    # text_lang = 'en'
    # myspeech = gTTS(text=label, lang=text_lang, slow=True)
    # myspeech.save("moo2.mp3")
    # os.system("moo2.mp3")

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()