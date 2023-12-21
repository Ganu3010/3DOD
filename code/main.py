from flask import Flask, request, jsonify
from models.pnpp import * 
from txt_to_npy import txt_to_npy
from data_loader import *
from flask_cors import CORS 
import os
import torch

app = Flask(__name__)
model = get_model(13)
checkpoint = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)

def classify(img, num_point = 4096):
    data = ScannetDatasetWholeScene([img])
    batches = len(data)
    scene_data, scene_label, scene_smpw, scene_point_index = data[i]
    num_blocks = scene_data.shape[0]
    print(num_blocks)
    batch_data = scene_data[0:1, ...]
    batch_label = scene_label[0:1, ...]
    batch_point_index = scene_point_index[0:1, ...]
    batch_smpw = scene_smpw[0:1, ...]
    batch_data[:, :, 3:6] /= 1.0
    torch_data = torch.Tensor(batch_data)
    torch_data = torch_data.float()
    torch_data = torch_data.transpose(2, 1)
    seg_pred, _ = model(torch_data)
    seg_pred = seg_pred.contiguous().data.max(2)[1].numpy()
    return seg_pred


@app.route('/', methods=  ['GET'])
def home():
    return jsonify([])

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':    
    
        img = request.files['file']
        result = classify(img)
    
        return jsonify(result), 201
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
        return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})


if __name__=="__main__":
    app.run(debug=True)