import open3d as o3d
import numpy as np
import cv2
import glob
import tqdm
import imageio

from pipeline import pipeline,pipeline_mask

if __name__ == "__main__":
    ## Read all calib files, images and point cloud files
    calib_files = sorted(glob.glob("../data/calib/*.txt"))
    scenario_images = sorted(glob.glob("../data/scenario/images/*.png"))
    scenario_points = sorted(glob.glob("../data/scenario/points/*.pcd"))
    image = cv2.imread(scenario_images[0])

    ## Create a video writer object
    output_handle = cv2.VideoWriter("../output/3d_to_2d_projection.avi",
                                    cv2.VideoWriter_fourcc(*'DIVX'),
                                    fps=15,
                                    frameSize=(image.shape[1], image.shape[0]))
    
    ## Create a progress bar
    pbar = tqdm.tqdm(total=len(scenario_images), position=0, leave=True)
    
    image_list = list()
    for im, pcd in zip(scenario_images, scenario_points):
        image = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
        point_cloud = o3d.io.read_point_cloud(pcd)
        # processed_image = pipeline(image, point_cloud, calib_files[0])
        processed_image = pipeline_mask(image, point_cloud, calib_files[0])
        
        image_list.append(processed_image)
        output_handle.write(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        pbar.update(1)

    output_handle.release()

    ## Write a GIF
    imageio.mimsave("../output/early_fusion_seg.gif", image_list, duration=5)

