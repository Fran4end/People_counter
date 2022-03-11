import cv2
import cvlib as cv
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox

#loading image
img = cv2.imread(".\\A.jpg");

#Detect objects

box, label, count = cv.detect_common_objects(img)
output = draw_bbox(img, box, label, count)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(output)
plt.show()
persone = 0
for obj in label:
    if(obj == 'person'):
        persone += 1


print (persone)