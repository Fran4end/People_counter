import math
import cv2
import numpy as np
import cvlib as cv
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox
thres = 0.45

img = cv2.imread('Noi_ma_piu_in_grande.jpg')
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
 
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

### rilevazione nei due modi ###
classIds, confs, bbox = net.detect(img, confThreshold=thres)
box, label, count = cv.detect_common_objects(img)

Nperson = 0;
for classId, confidence,box in zip(classIds.flatten(), confs.flatten(), bbox):
    if(classNames[classId-1] == 'person'):
        Nperson += 1
        #### disegna il rettangolo con il nome dell'oggeto ###
        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
        cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

persone = 0
for obj in label:
    if(obj == 'person'):
        persone += 1

### disegna il rettangolo con il nome dell'oggeto ###
# output = draw_bbox(img, box, label, count)
# output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 10))
# plt.axis("off")
# #plt.imshow(output)
# #plt.show()

Nperson = round(((Nperson + persone)/2), 0)

cv2.imshow('Output', img)
print(Nperson)
cv2.waitKey(0)