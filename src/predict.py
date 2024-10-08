import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt  # Importing matplotlib for visualization

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

def predict_zone_7_center(model_path, scaler_path, previous_zones_data):
    model = torch.load(model_path)  
    model.eval()
    
    scaler_X, scaler_y = load_scalers(scaler_path)
    input_data = np.array(previous_zones_data).reshape(1, -1)
    input_data_scaled = scaler_X.transform(input_data)
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        predicted_scaled = model(input_tensor).numpy()
    predicted = scaler_y.inverse_transform(predicted_scaled)
    
    return predicted[0] / 100

def visualize_zones_fixed_center(previous_zones_data, actual_zone_7, predicted_zone_7):
    plt.figure(figsize=(10, 10))
    
    # Fixed radii for zones
    zone_7_radius_meters = 200 
    zone_6_radius_meters = 325 
    
   
    actual_zone_6 = (previous_zones_data[-3] / 100, previous_zones_data[-2] / 100)  # Convert to meters
    
    actual_zone_6_circle = plt.Circle((actual_zone_6[0], actual_zone_6[1]), zone_6_radius_meters, color='green', fill=False, linestyle='-', label='Actual Zone 5')
    plt.gca().add_patch(actual_zone_6_circle)
    plt.scatter(actual_zone_6[0], actual_zone_6[1], color='green', marker='o', label='Actual Zone 5 Center')
    
    actual_zone_7_circle = plt.Circle((actual_zone_7[0], actual_zone_7[1]), zone_7_radius_meters, color='blue', fill=False, label='Actual Zone 6')
    plt.gca().add_patch(actual_zone_7_circle)
    plt.scatter(actual_zone_7[0], actual_zone_7[1], color='blue', label='Actual Zone 6 Center')
    
    predicted_zone_7_circle = plt.Circle((predicted_zone_7[0], predicted_zone_7[1]), zone_7_radius_meters, color='red', fill=False, linestyle='--', label='Predicted Zone 6')
    plt.gca().add_patch(predicted_zone_7_circle)
    plt.scatter(predicted_zone_7[0], predicted_zone_7[1], color='red', marker='x', label='Predicted Zone 6 Center')
    

    plt.xlim(-1500, 1500)  
    plt.ylim(-1500, 1500)  
    
    plt.title('Zone Prediction')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Example usage
previous_zones_data = [
    -24163.79, -16028.86, -759.41, 
    -34527.38, -38057.36, -759.41, 
    -54122.96, -47650.58, -759.41, 
    -60620.15, -40702.24, -759.41,  
    -56871.62, -29192.12, -759.41  
]

predicted_zone_7_center = predict_zone_7_center('data/zone_prediction_model.pth', 'data/scalers.pkl', previous_zones_data)
actual_zone_7_center = [-866.1951, -422.8078, -7.5941]

visualize_zones_fixed_center(previous_zones_data, actual_zone_7_center, predicted_zone_7_center)
