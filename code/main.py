from flask import Flask, request, jsonify, send_from_directory, render_template
from models.pnpp import * 
from data_loader import *
import os
import torch
from tqdm import tqdm
import open3d as o3d


g_class2color = {0: [0, 255, 0], 1: [0, 0, 255], 2: [0, 255, 255], 3: [255, 255, 0], 
                 4: [255, 0, 255], 5: [100, 100, 255], 6: [200, 200, 100], 7: [170, 120, 200],
                 8: [255, 0, 0], 9: [200, 100, 100], 10: [10, 200, 100], 11: [200, 200, 200], 
                 12: [50, 50, 50]}
model = get_model(13)
checkpoint = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def add_vote(vote_label_pool, point_idx, pred_label, weight):       # helper function for predictions
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def to_pcd(ip_file):                                                # helper function for predictions
    if ip_file.split('.')[-1] == 'pcd':
        return ip_file
    output_file = ip_file.split('.')[0] + '.pcd'    
    points = np.loadtxt(ip_file) if ip_file.split('.')[-1] == 'txt' else np.load(ip_file)
    op = o3d.geometry.PointCloud()
    points[:, -3:] /= 255
    xyz_min = np.amin(points, axis=0)[:3]
    points[:, :3] -= xyz_min
    op.points = o3d.utility.Vector3dVector(points[:, :3])
    op.colors = o3d.utility.Vector3dVector(points[:, -3:])
    o3d.io.write_point_cloud(output_file, op)
    return output_file        

def classify(img, num_point = 4096, num_votes = 5):                 # Classification function.
    filename = 'static/input/'+img.filename.split('.')[0]+'.pcd'
    if os.path.exists(filename):
        return filename
    else: 
        filename = filename.split('.')[0]+'.txt'
    data = DataLoader([img])
    for batch_idx in range(len(data)):
        whole_scene_data = data.scene_points_list[batch_idx]
        whole_scene_label = data.semantic_labels_list[batch_idx]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], 13))

        for _ in range(num_votes):
            scene_data, scene_label, scene_smpw, scene_point_index = data[batch_idx]
            num_blocks = scene_data.shape[0]
            batch_data = np.zeros((1, num_point, 9))

            batch_label = np.zeros((1, num_point))
            batch_point_index = np.zeros((1, num_point))
            batch_smpw = np.zeros((1, num_point))
            with torch.no_grad():
                for block in tqdm(range(num_blocks), total = num_blocks):
                    end_idx = min((block+1), num_blocks)
                    real_batch_size = end_idx - block
                    batch_data[0:real_batch_size, ...] = scene_data[block:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[block:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[block:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[block:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0
                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = model(torch_data)
                    batch_pred_label = seg_pred.contiguous().data.max(2)[1].numpy()
                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...], batch_pred_label[0:real_batch_size, ...], batch_smpw[0:real_batch_size, ...])
        pred_label = np.argmax(vote_label_pool, 1)
        
    with open(filename, 'w') as f:
        for i in range(len(whole_scene_label)):
            color = g_class2color[pred_label[i]]
            f.write('{} {} {} {} {} {} \n'.format(whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1], color[2]))

    return filename

app = Flask(__name__, static_url_path='/static')                    # Flask App.
UPLOAD_FOLDER = 'static/input/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'npy', 'pcd'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def index():
    return render_template('Home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():                                                      # Predict route
    if request.method == 'POST':    
        file = request.files['file']
        result = classify(file, num_votes=1)
        path_name = to_pcd(result)
        file_name = path_name.split('/')[-1]

        o3d.visualization.webrtc_server.enable_webrtc()
        point_cloud = o3d.io.read_point_cloud(path_name)
        # Compute the axis-aligned bounding box (AABB) of the point cloud
        aabb = point_cloud.get_axis_aligned_bounding_box()
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        file.save('static/output/'+file_name)
        o3d.visualization.draw_geometries([point_cloud, aabb])
        # return render_template('pcd.html', file_name = file_name, path_name = path_name)
    
    else:
        return jsonify([]), 404

@app.route('/upload', methods=['POST'])
def upload_file():                                                  # Upload route
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return jsonify(success=True, message="Point cloud processed successfully", output_path='static/input/')


@app.route('/visualize', methods=['POST'])
def visualize():
    if request.method == 'POST':
        file = request.files['file']
        path_name = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        path_name = to_pcd(path_name)
        file_name = path_name.split('/')[-1]

        o3d.visualization.webrtc_server.enable_webrtc()
        point_cloud = o3d.io.read_point_cloud(path_name)
        # Compute the axis-aligned bounding box (AABB) of the point cloud
        aabb = point_cloud.get_axis_aligned_bounding_box()
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        o3d.visualization.draw_geometries([point_cloud, aabb])
        # return render_template('pcd.html', file_name = file_name, path_name = path_name)
    else:
        return jsonify([]), 404
    

 
if __name__=="__main__":
    app.run(debug=True)
