# import open3d as o3d

# pcd = o3d.io.read_point_cloud('predictions/output_4.pcd')

# o3d.visualization.draw_geometries([pcd])


from flask import Flask, request
app = Flask(__name__,static_url_path='/static')
if __name__ == "__main__":
    app.run(debug = True)