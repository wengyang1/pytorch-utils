import cv2
from PIL import Image
import numpy as np

gt_path = '/Users/wengyang/codes/LapDepth/datasets/KITTI/data_depth_annotated/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png'
rgb_path = '/Users/wengyang/codes/LapDepth/datasets/KITTI/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000086.png'

gt_image = cv2.imread(gt_path)
print(gt_image.shape,type(gt_image))
gt = Image.open(gt_path)
print(gt.size,type(gt))
gt = np.asarray(gt, dtype=np.float32) / 255.0


grb_image = cv2.imread(rgb_path)
print(grb_image.shape,type(grb_image))
rgb = Image.open(rgb_path)
print(rgb.size,type(rgb))
rgb = np.asarray(rgb, dtype=np.float32) / 255.0
print()


path = '/Users/wengyang/codes/LapDepth/out_kitti_demo.jpg'
image = cv2.imread(path)
print(image.shape,type(image))
image = Image.open(path)
print()