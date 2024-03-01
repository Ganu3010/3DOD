import open3d as o3d
import numpy as np
import os
from main import classify, to_pcd

for file in os.listdir('static/output/'):
    pcd = o3d.io.read_point_cloud('static/output/'+file)
    o3d.visualization.draw_geometries([pcd], window_name = file)