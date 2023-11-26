from flask import Flask, request, jsonify
from models.pnpp import * 
import torch
app = Flask(__name__)
model = get_model(13)
checkpoint = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def transform_image(img):
    return torch.rand(6, 9, 2048)

@app.route('/', methods=  ['GET'])
def home():
    return jsonify([])

@app.route('/predict', methods=['POST'])
def predict():
    img = request.form['file']
    xyz = transform_image(img)
    result = model(xyz)    
    
    return jsonify(result)
    

if __name__=="__main__":
    app.run(debug=True)

