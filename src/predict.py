import numpy as np
import torch
import torch.nn as nn
import pickle

# Function to load the scalers
def load_scalers(scaler_path='data/scalers.pkl'):
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers['scaler_X'], scalers['scaler_y']

# Define the neural network model
class ZonePredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(ZonePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Function to predict the zone 7 center coordinates using previous zone data
def predict_zone_7_center(model_path, scaler_path, previous_zones_data):
    # Load the model and scalers
    model = ZonePredictionModel(len(previous_zones_data))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    scaler_X, scaler_y = load_scalers(scaler_path)
    
    # Prepare the input data
    input_data = np.array(previous_zones_data).reshape(1, -1)
    input_data_scaled = scaler_X.transform(input_data)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    
    # Predict the next zone center
    with torch.no_grad():
        predicted_scaled = model(input_tensor).numpy()
    predicted = scaler_y.inverse_transform(predicted_scaled)
    
    return predicted[0]

previous_zones_data = [
    -11384.89, 27641.88, -759.41, 
    -24164.64, 49101.76, -759.41, 
    -36882.68, 70611.21, -759.41, 
    -41927.24, 76703.06, -759.41,  
    -59019.12, 87003.52, -759.41  
]

predicted_zone_7_center = predict_zone_7_center('data/zone_prediction_model.pth', 'data/scalers.pkl', previous_zones_data)
print(f"Predicted Zone 6 Center: x={predicted_zone_7_center[0]}, y={predicted_zone_7_center[1]}, z={predicted_zone_7_center[2]}")
