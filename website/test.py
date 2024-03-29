import numpy as np
import os 
import open3d as o3d
from utils import get_bounding_boxes, visualize_bounding_boxes

if __name__=='__main__':
    visualize_bounding_boxes('static/output/scene0000_00_vh_clean_2/output.ply')
    get_bounding_boxes('static/output/scene0707_00_vh_clean_2')