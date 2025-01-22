import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt
import time
import statistics
import random
from ultralytics import YOLO

from lidar_to_camera_projection import LiDARtoCamera
from object_detection_2d import run_obstacle_detection

yolo = YOLO('yolov5su.pt')


class FusionLidarCamera(LiDARtoCamera):
    def rectContains(self, rect, pt, w, h, shrink_factor=0):
        # x1 = int(rect[0] * w - rect[2] * w * 0.5 * (1 - shrink_factor))  # center_x - width /2 * shrink_factor
        # y1 = int(rect[1] * h - rect[3] * h * 0.5 * (1 - shrink_factor))  # center_y - height /2 * shrink_factor
        # x2 = int(rect[0] * w + rect[2] * w * 0.5 * (1 - shrink_factor))  # center_x + width/2 * shrink_factor
        # y2 = int(rect[1] * h + rect[3] * h * 0.5 * (1 - shrink_factor))  # center_y + height/2 * shrink_factor
        x1 = int(rect[0])  # center_x - width /2 * shrink_factor
        y1 = int(rect[1])  # center_y - height /2 * shrink_factor
        x2 = int(rect[2])  # center_x + width/2 * shrink_factor
        y2 = int(rect[3])  # center_y + height/2 * shrink_factor


        return x1 < pt[0] < x2 and y1 < pt[1] < y2

    def filter_outliers(self, distances):
        inliers = []
        mu = statistics.mean(distances)
        std = statistics.stdev(distances)
        for x in distances:
            if abs(x - mu) < std:
                inliers.append(x)
        return inliers

    def get_best_distance(self, distances, technique="closest"):
        if technique == "closest":
            return min(distances)
        elif technique == "average":
            return statistics.mean(distances)
        elif technique == "random":
            return random.choice(distances)
        else:
            return statistics.median(sorted(distances))

    def lidar_camera_fusion(self, pred_bboxes, image):
        img_bis = image.copy()
        # 使用 matplotlib.pyplot 获取颜色映射，避免导入不存在的模块
        cmap = plt.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        distances = []
        for box in pred_bboxes:
            distances = []
            # if self.pcd_img_points.shape[0] == 0:
            #     print(pcd_img_points,"为空")
            #     continue
         
            for i in range(self.pcd_img_points.shape[0]):
                # depth = self.imgfov_pc_rect[i, 2]
                depth = self.pcd_points_in_img[i, 0]
                if (self.rectContains(box, self.pcd_img_points[i], image.shape[1], image.shape[0], shrink_factor=0.2) == True):
                    
                    distances.append(depth)
                    color = cmap[int(510.0 / depth), :]
                    # 修正 cv2.circle 函数的调用，正确匹配括号
                    cv2.circle(img_bis, (int(np.round(self.pcd_img_points[i, 0])), int(np.round(self.pcd_img_points[i, 1]))), 2,tuple(color), thickness=-1)
            h, w, _ = img_bis.shape
            if (len(distances) > 2):
                distances = self.filter_outliers(distances)
                best_distance = self.get_best_distance(distances, technique="average")
                cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA)
            distances_to_keep = []
        cv2.imwrite("../output/final_result_det.png", img_bis)
        return img_bis, distances

if __name__ == "__main__":
    idx = 0

    ## Load the list of files
    calib_files = sorted(glob.glob("../data/calib/*.txt"))
    pointcloud_files = sorted(glob.glob("../data/velodyne/*.pcd"))
    image_files = sorted(glob.glob("../data/img/*.png"))

    ## Read the image file
    image = cv2.imread(image_files[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ## Read Point cloud files
    pcd = o3d.io.read_point_cloud(pointcloud_files[idx])
    points_pcd = np.asarray(pcd.points)

    ## Convert from LiDAR to Camera coord
    lidar2cam = FusionLidarCamera(calib_files[idx])

    ## Point cloud data in image
    image_pcd = lidar2cam.show_pcd_on_image(image.copy(), points_pcd)

    ## Object detection 2d
    results = yolo(image)

    pred_bboxes = []
    for r in results:
        boxes = r.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            conf = box.conf[0]
            cls = box.cls[0].astype(int)
            if conf > 0.4:  # 可根据实际调整置信度阈值
                pred_bboxes.append([x1, y1, x2, y2, conf, cls])

    for box in pred_bboxes:
        x1, y1, x2, y2 = box[0:4]
        cv2.rectangle(image_pcd, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    lidar_img_with_bboxes = image_pcd
    # print(pred_bboxes)
    cv2.imwrite("../output/fig_fusion1.png", lidar_img_with_bboxes)
    # fig_fusion = plt.figure(figsize=(14, 7))
    # ax_fusion = fig_fusion.add_subplot(111)
    # ax_fusion.imshow(lidar_img_with_bboxes)
    
    # plt.show()
    result, pred_bboxes = run_obstacle_detection(image)
    final_result, _ = lidar2cam.lidar_camera_fusion(pred_bboxes, result)
    # cv2.imwrite("../output/final_result_det.png", final_result)
    # fig_keeping = plt.figure(figsize=(14, 7))
    # ax_keeping = fig_keeping.add_subplot(111)
    # ax_keeping.imshow(final_result)
    
    # plt.show()