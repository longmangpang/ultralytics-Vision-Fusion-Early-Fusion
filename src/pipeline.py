import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt
# import tensorflow as tf
import time
import statistics
import random
from ultralytics import YOLO

from lidar_to_camera_projection import LiDARtoCamera
from object_detection_2d import run_obstacle_detection
from fuse_point_clouds_with_masks import FusionLidarCamera


def pipeline(image, point_cloud, calib_file):
    "For a pair of 2 Calibrated Images"
    img = image.copy()

    lidar2cam = FusionLidarCamera(calib_file)
    # Show LidAR on Image
    lidar_img = lidar2cam.show_pcd_on_image(image, np.asarray(point_cloud.points))
    # Load YOLOv5.7 model
    yolo = YOLO('yolov5s.pt')
    # Run obstacle detection in 2D
    results = yolo(img)
    pred_bboxes = []
    for r in results:
        boxes = r.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = box.cls[0].astype(int)
            if conf > 0.4:  # 可根据实际调整置信度阈值
                pred_bboxes.append([x1, y1, x2, y2, conf, cls])
    # Fuse Point Clouds & Bounding Boxes
    img_final, _ = lidar2cam.lidar_camera_fusion(pred_bboxes, lidar_img)
    return img_final

def pipeline_mask(image, point_cloud, calib_file):
    "For a pair of 2 Calibrated Images"
    img = image.copy()

    
    
    # Load YOLOv5.7 model
    yolo = YOLO('yolov8n-seg.pt')
    # Run obstacle detection in 2D
    results = yolo(img)
    yolo_classes = list(yolo.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
    lidar2cam = FusionLidarCamera(calib_file,colors)
    # Show LidAR on Image
    lidar_img = lidar2cam.show_pcd_on_image(img, np.asarray(point_cloud.points))
    # Fuse Point Clouds & Bounding Boxes
    img_final, _ = lidar2cam.lidar_camera_fusion(results,image)

    return img_final

def run_obstacle_detection(img):
    # Load YOLOv5.7 model
    model = YOLO('yolov5s.pt')
    start_time = time.time()
    # 将图像从 BGR 格式转换为 RGB 格式（如果输入图像是 BGR 格式的话）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用 YOLOv5.7 进行预测，这里返回的结果包含了检测到的目标信息，如边界框、类别、置信度等
    results = model(img)
    pred_bboxes = []
    result = img.copy()
    for r in results:
        boxes = r.boxes.cpu().numpy()  # 获取检测到的边界框信息并转为 numpy 数组方便后续操作
        for box in boxes:
            # 获取边界框坐标（x1, y1, x2, y2）、类别、置信度等信息
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = box.cls[0].astype(int)
            # 这里可以根据自己的需求添加筛选条件，比如置信度阈值等，示例中简单打印相关信息
            if conf > 0.4:  # 假设置信度阈值设为 0.4
                pred_bboxes.append([x1, y1, x2, y2, conf, cls])
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在图像上绘制边界框
    exec_time = time.time() - start_time
    print(f"Execution time: {exec_time} seconds")
    return result, np.array(pred_bboxes)


if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("../data/img/000031.png"), cv2.COLOR_BGR2RGB)
    pcd = o3d.io.read_point_cloud("../data/velodyne/000031.pcd")
    calib_file = "../data/calib/000031.txt"

    result_img = pipeline(image, pcd, calib_file)

    fig = plt.figure(figsize=(14, 7))
    ax_keeping = fig.subplots()
    ax_keeping.imshow(result_img)
    cv2.imwrite("../output/result_img.png", result_img)
    plt.show()