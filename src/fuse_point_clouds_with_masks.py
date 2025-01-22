import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt
import statistics
import random
from ultralytics import YOLO

from lidar_to_camera_projection import LiDARtoCamera


class FusionLidarCamera(LiDARtoCamera):
    def __init__(self, calib_file,colors):
        super().__init__(calib_file)
        self.colors = colors
      

    def rectContains(self, mask, pt):
        # 确保 rect 中的元素可以转换为整数
        contour = np.int32(mask)  # 轮廓点
        point = (pt[0], pt[1])  # 你要判断的点
        return cv2.pointPolygonTest(contour, point, False) == 1

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

    def lidar_camera_fusion(self, results, image):
        
        img_bis = image.copy()
        cmap = plt.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        distances = []
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                mask_distances = []  # 存储当前 mask 内的距离
                for i in range(self.pcd_img_points.shape[0]):
                    depth = self.pcd_points_in_img[i, 0]
                    if (self.rectContains(mask, self.pcd_img_points[i]) == True):
                        mask_distances.append(depth)
                        color = cmap[int(510.0 / depth), :]
                        cv2.circle(img_bis, (int(np.round(self.pcd_img_points[i, 0])), int(np.round(self.pcd_img_points[i, 1]))), 2, tuple(color), thickness=-1)
                h, w, _ = img_bis.shape
                if (len(mask_distances) > 2):
                    mask_distances = self.filter_outliers(mask_distances)
                    best_distance = self.get_best_distance(mask_distances, technique="average")
                    center_x = int(np.mean(mask[:, 0]))
                    center_y = int(np.mean(mask[:, 1]))
                    cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA)
                points = np.int32([mask])
                # 获取类别索引
                color_number = int(box.cls[0])        
                # 随机生成颜色
                color = self.colors[color_number]
                #cv2.fillPoly(img, points, colors[color_number])
                # 绘制轮廓
                cv2.polylines(img_bis, [points], isClosed=True, color=color, thickness=2)
                distances.extend(mask_distances)  # 将当前 mask 的距离添加到总距离列表中
        return img_bis, distances


if __name__ == "__main__":
    idx = 0

    calib_files = sorted(glob.glob("../data/calib/*.txt"))
    pointcloud_files = sorted(glob.glob("../data/velodyne/*.pcd"))
    image_files = sorted(glob.glob("../data/img/*.png"))

    image = cv2.imread(image_files[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pcd = o3d.io.read_point_cloud(pointcloud_files[idx])
    points_pcd = np.asarray(pcd.points)

    lidar2cam = FusionLidarCamera(calib_files[idx])
    image_pcd = lidar2cam.show_pcd_on_image(image.copy(), points_pcd)

    yolo = YOLO('yolov8n-seg.pt')
    results = yolo(image)
    yolo_classes = list(yolo.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    colors = [random.choices(range(256), k=3) for _ in classes_ids]

    # pred_masks = []
    

   
    lidar_img_with_masks = image_pcd

    
    cv2.imwrite("../output/fig_fusion1.png", lidar_img_with_masks)

    final_result, _ = lidar2cam.lidar_camera_fusion(results, image)
    cv2.imwrite("../output/final_result_seg.png", final_result)