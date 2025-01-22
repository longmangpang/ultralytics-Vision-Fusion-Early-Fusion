import numpy as np
import open3d as o3d
import glob
import cv2
import matplotlib.pyplot as plt

# 定义一个名为LiDARtoCamera的类，用于处理激光雷达点云数据到相机坐标的转换以及相关操作
class LiDARtoCamera():
    """
    This will read the calibration file, and create the transformation matrix from one
    co-ordinate frame to another one
    这个类将读取校准文件，并创建从一个坐标系统到另一个坐标系统的转换矩阵
    """
    def __init__(self, calibfile):
        """
        类的初始化函数，在创建类的实例时调用，用于读取校准文件并提取相关参数

        :param calibfile: 校准文件的路径，字符串类型
        """
        calib_data = self.read_calibfile(calibfile)
        ## Intrinsics param, the images are belongs to P2
        # 相机内参矩阵，这里对应的图像属于P2（可能是某种相机模型下的参数标识）
        self.P = calib_data["P2"].reshape((3, 4))
        ## Rotation from reference camera coord to rect camera coord
        # 从参考相机坐标到校正后相机坐标的旋转矩阵
        self.R0 = calib_data["R0_rect"].reshape((3, 3))
        ## Translation from velodyne coord to reference camera coord
        # 从激光雷达坐标到参考相机坐标的平移矩阵
        self.V2C = calib_data["Tr_velo_to_cam"].reshape((3, 4))

    def read_calibfile(self, filepath):
        """
        读取校准文本文件，并将其内容转换为字典形式的函数

        :param filepath: 校准文件的路径，字符串类型
        :return data: 字典类型，包含校准文件中的键值对信息，键为参数名称，值为对应的参数数值（以numpy数组形式存储）
        """
        data = dict()
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if (len(line) == 0):
                    continue
                key, value = line.split(":", 1)

                data[key] = np.array([float(x) for x in value.split()])
        return data
    
    def project_pcd_to_image(self, pcd_points):
        """
        将所有点从激光雷达坐标转换为图像坐标的函数
        使用公式：2-D points = P * R0 * R|t * 3-D points

        :param pcd_points: 点云的点坐标（x, y, z），二维numpy数组，形状为 (n x 3)，n为点的数量
        :return img_points: 转换后的图像坐标（x, y），二维numpy数组，形状为 (n x 2)
        """
        ## Homogeneous conversion of the matrices to compatable shape for matrix multiplication
        ## 为了进行矩阵乘法，对相关矩阵进行齐次坐标转换，使其维度匹配以便计算
        ## 2-D points = P   * R0  * R|t * 3-D points
        ## (3xn)      = (3x4) (3x3) (3x4) (3xn) convert into
        ## (3xn)      = (3x4) (4x4) (4x4) (4xn)
        P = self.P 
        # 将旋转矩阵R0扩充为4x4的齐次坐标形式，在下方添加一行[0., 0., 0., 1.]
        R0 = np.vstack([np.hstack([self.R0, [[0.], [0.], [0.]]]), [0., 0., 0., 1.]])
        # 将平移矩阵V2C扩充为4x4的齐次坐标形式，在下方添加一行[0., 0., 0., 1.]
        Rt = np.vstack([self.V2C, [0., 0., 0., 1.]])

        ## Homogeneous convertion of the input points dimension from nx3 to 4xn 
        # 将输入的点云坐标维度从 (nx3) 转换为 (4xn) 的齐次坐标形式，先转置再添加一行全1的元素（代表齐次坐标的w分量）
        pcd_points = np.transpose(pcd_points)
        pcd_points = np.vstack([pcd_points, np.ones(pcd_points.shape[1])])

        ## Applying the formula
        # 按照坐标转换公式，依次进行矩阵乘法运算
        img_points = np.dot(P, R0)
        img_points = np.dot(img_points, Rt)
        img_points = np.dot(img_points, pcd_points)

        ## Homogeneous to Euclidean conversion
        # 将齐次坐标形式的点转换回欧几里得坐标形式（除以w分量），先转置，再分别对x、y坐标除以z坐标（即w分量）
        img_points = np.transpose(img_points)
        img_points[:, 0] /= img_points[:, 2]
        img_points[:, 1] /= img_points[:, 2]

        return img_points[:, :2]
    
    def get_pcd_in_image_fov(self, pcd_points, xmin, xmax, ymin, ymax, clip_dist=2.0):
        """
        从点云文件中筛选出仅在图像视场内的点的函数

        :param pcd_points: 点云文件中的点坐标，二维numpy数组
        :param xmin: 图像x轴最小值，整数类型，这里通常为0
        :param xmax: 图像x轴最大值，整数类型，通常为图像的宽度
        :param ymin: 图像y轴最小值，整数类型，这里通常为0
        :param ymax: 图像y轴最大值，整数类型，通常为图像的高度
        :param clip_dist: 激光雷达的最小裁剪距离，浮点数类型，用于过滤距离过近的点
        :return pcd_points_in_img: 筛选后在图像视场内的点云文件中的点坐标，二维numpy数组
        :return img_points: 点云在相机坐标下的坐标，二维numpy数组
        :return fov_idx: 一个布尔类型的掩码数组，用于筛选图像视场内的点
        """
        img_points = self.project_pcd_to_image(pcd_points)
        # 通过逻辑与操作构建掩码，筛选出x坐标在图像范围内、y坐标在图像范围内且激光雷达点的x坐标大于裁剪距离的点
        fov_idx = ((img_points[:, 0] >= xmin)&
                   (img_points[:, 0] < xmax)&
                   (img_points[:, 1] >= ymin)&
                   (img_points[:, 1] < ymax))
        fov_idx = fov_idx & (pcd_points[:, 0] > clip_dist)
        pcd_points_in_img = pcd_points[fov_idx, :]
        return pcd_points_in_img, img_points, fov_idx
    
    def show_pcd_on_image(self, image, pcd_points):
        """
        在给定图像上绘制点云点的函数

        :param image: cv2图像，输入的图像数据
        :param pcd_points: 点云的点坐标，二维numpy数组
        :return image: cv2图像，在传入的图像上绘制了点云点后的图像
        """
        self.pcd_points_in_img, img_points, fov_idx = self.get_pcd_in_image_fov(pcd_points,
                                                                           xmin=0,
                                                                           xmax=image.shape[1],
                                                                           ymin=0,
                                                                           ymax=image.shape[0],
                                                                           clip_dist=2.0)
        self.pcd_img_points = img_points[fov_idx, :]

        ## Create a color map scale
        # 获取HSV颜色映射，生成256个颜色值，再将其转换为RGB格式并乘以255，得到取值范围在0-255的颜色数组，用于根据深度信息给点云点上色
        cmap = plt.colormaps.get_cmap("hsv")
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        
        ## Draws the PCD points on image
        # 循环遍历筛选后的点云点，根据点的深度信息获取对应的颜色，然后在图像上以圆形绘制每个点
        for i in range(self.pcd_img_points.shape[0]):
            depth = self.pcd_points_in_img[i, 0]
            color = cmap[int(2*255.0/depth), :]
            cv2.circle(image, 
                       (int(np.round(self.pcd_img_points[i, 0])), int(np.round(self.pcd_img_points[i, 1]))),
                       radius=3,
                       color=tuple(color),
                       thickness=-1)

        return image

if __name__ == "__main__":
    idx = 0

    ## Load the list of files 
    # 获取校准文件列表，按照文件名排序，这里假设校准文件存放在../data/calib/目录下，且文件后缀为.txt
    calib_files = sorted(glob.glob("../data/calib/*.txt"))
    # 获取点云文件列表，按照文件名排序，假设点云文件存放在../data/velodyne/目录下，且文件后缀为.pcd
    pointcloud_files = sorted(glob.glob("../data/velodyne/*.pcd"))
    # 获取图像文件列表，按照文件名排序，假设图像文件存放在../data/img/目录下，且文件后缀为.png
    image_files = sorted(glob.glob("../data/img/*.png"))
    
    ## Read the image file
    # 读取指定索引的图像文件，读取后的图像格式为BGR，需要后续转换为RGB格式
    image = cv2.imread(image_files[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ## Read Point cloud files
    # 读取指定索引的点云文件，将其转换为open3d的点云数据类型，再进一步提取点坐标信息转换为numpy数组
    pcd = o3d.io.read_point_cloud(pointcloud_files[idx])
    points_pcd = np.asarray(pcd.points)

    ## Convert from LiDAR to Camera coord
    # 创建LiDARtoCamera类的实例，传入对应的校准文件，用于后续的坐标转换和相关操作
    lidar2cam = LiDARtoCamera(calib_files[idx])

    image = lidar2cam.show_pcd_on_image(image.copy(), points_pcd)
    
    ## Save the image
    # 设置图像显示的尺寸大小，展示绘制了点云点后的图像，并将图像保存到指定路径下
    plt.figure(figsize=(14, 7))
    plt.imshow(image)
    plt.show() 
    # plt.savefig("../output/000031.png")
    # cv2.imwrite("../output/000031_1.png",image)