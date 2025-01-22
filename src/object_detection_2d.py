import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from ultralytics import YOLO


model = YOLO('yolov5s.pt')


def run_obstacle_detection(img):
    start_time = time.time()
    # 将图像从BGR格式转换为RGB格式（如果输入图像是BGR格式的话，这里和原代码类似操作）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用YOLOv5进行预测，这里返回的结果包含了检测到的目标信息，如边界框、类别、置信度等
    results = model(img)
    pred_bboxes = []
    result = img.copy()
    for r in results:
        boxes = r.boxes.cpu().numpy()  # 获取检测到的边界框信息并转为numpy数组方便后续操作
        for box in boxes:
            # 获取边界框坐标（x1, y1, x2, y2）、类别、置信度等信息
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = box.cls[0].astype(int)
            # 这里可以根据自己的需求添加筛选条件，比如置信度阈值等，示例中简单打印相关信息
            if conf > 0.4:  # 假设置信度阈值设为0.4
                pred_bboxes.append([x1, y1, x2, y2, conf, cls])
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在图像上绘制边界框
    exec_time = time.time() - start_time
    print(f"Execution time: {exec_time} seconds")
    return result, np.array(pred_bboxes)


if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("../data/img/000031.png"), cv2.COLOR_BGR2RGB)
    result, pred_bboxes = run_obstacle_detection(image)
    cv2.imwrite("../output/2d_object_detection1.png",result)
    # fig_camera = plt.figure(figsize=(14, 7))
    # ax_lidar = fig_camera.add_subplot(111)
    # ax_lidar.imshow(result)
    # plt.savefig("../output/2d_object_detection.png")
    # plt.show()