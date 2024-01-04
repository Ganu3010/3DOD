from flask import Flask, request, jsonify, send_from_directory
from models.pnpp import * 
from txt_to_npy import txt_to_npy
from data_loader import *
from flask_cors import CORS 
import os
import torch
from tqdm import tqdm

g_class2color = {0: [0, 255, 0], 1: [0, 0, 255], 2: [0, 255, 255], 3: [255, 255, 0], 
                 4: [255, 0, 255], 5: [100, 100, 255], 6: [200, 200, 100], 7: [170, 120, 200],
                 8: [255, 0, 0], 9: [200, 100, 100], 10: [10, 200, 100], 11: [200, 200, 200], 
                 12: [50, 50, 50]}
app = Flask(__name__)
model = get_model(13)
checkpoint = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


ALLOWED_EXTENSIONS = {'npy'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool
    
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)


def classify(img, num_point = 4096, num_votes = 5):
    data = ScannetDatasetWholeScene([img])
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
            print(num_blocks)
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
        filename = 'predictions/output_{}.txt'.format(1)
        with open(filename, 'w') as f:
            for i in range(len(whole_scene_label)):
                color = g_class2color[pred_label[i]]
                f.write('{} {} {} {} {} {}'.format(whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1], color[2]))
        
    return filename.split('/')[1]


@app.route('/', methods=  ['GET'])
def home():
    return jsonify([])


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':    
    
        file = request.files['file']
        print(file)
        print("file name "+file.filename)
        result = classify(file)
        return send_from_directory('predictions/', result), 201
    else:
        return jsonify([]), 404



@app.route('/upload', methods=['POST'])


def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        #conversion from .txt to .pcd
        # Process the point cloud
        #output_path = process_point_cloud_file(file_path)
      
        #return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})
        return jsonify(success=True, message="Point cloud processed successfully", output_path=output_path)


# def process_point_cloud_file(file_path):
#     # Read the point cloud from the input file
#     pcd = o3d.io.read_point_cloud(file_path, format="xyzrgb")

#     for point in pcd.colors:
#         point /= 255.0

#     # Save the processed point cloud to a new file
#     output_path = 'uploads/output.pcd'
#     o3d.io.write_point_cloud(output_path, pcd)

#     return output_path

if __name__=="__main__":
    app.run(debug=True)
