from models.pnpp import *
import torch

model = get_model(13)
checkpoint = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
op, _ = model(torch.rand(6, 9, 4096))
print(op.shape)