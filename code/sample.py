import open3d as o3d


# o3d.geometry.visualize('static/input/Area_1_conferenceRoom_2.pcd')
pcd = o3d.io.read_point_cloud('static/input/Area_1_conferenceRoom_2.pcd')

o3d.visualization.draw_geometries([pcd])