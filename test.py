import open3d as o3d

pcd = o3d.io.read_point_cloud('website/static/input/Area_1_hallway_4.ply')

# o3d.io.write_point_cloud('website/static/input/Area_1_hallway_4.ply', pcd)

o3d.visualization.draw_geometries([pcd])