import numpy as np
import os 
import open3d as o3d
from utils import get_bounding_boxes, get_coords_color
# from utils import COLOR_DETECTRON2
import random
import numpy as np
from tqdm import tqdm


if __name__=='__main__':
    # pcd = o3d.io.read_point_cloud('sample.ply')
    # print(len(pcd.points))
    get_bounding_boxes('/home/manas/test/3DOD/website/static/output/scene0006_00_vh_clean_2')