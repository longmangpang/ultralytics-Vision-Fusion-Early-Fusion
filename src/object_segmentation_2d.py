from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("yolo11n-seg.pt") 

img = cv2.imread("../data/img/000031.png")

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.2

results = model.predict(img, conf=conf)
colors = [random.choices(range(256), k=3) for _ in classes_ids]

for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        
         # 获取类别索引
        color_number = int(box.cls[0])        
        # 随机生成颜色
        color = colors[color_number]

        #cv2.fillPoly(img, points, colors[color_number])
        # 绘制轮廓
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)


# cv2.imshow("Image", img)
# cv2.waitKey(0)
cv2.imwrite("./result2.png", img)

