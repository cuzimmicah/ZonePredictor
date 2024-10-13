import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from manim import *
import numpy as np

"""
Plotting Loss Gradient
"""

class ZonePredictorLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=5, output_dim=2):
        super(ZonePredictorLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_zone_predictor_for_phase(phase_number, input_dim=2, hidden_dim=128, num_layers=4):
    return ZonePredictorLSTM(input_dim, hidden_dim, num_layers)

class ZoneDataset(Dataset):
    def __init__(self, data, phase_to_predict):
        self.data = []
        self.phase_to_predict = phase_to_predict
        
        for match in data:
            zones = match['zones']
            if len(zones) >= phase_to_predict:
                sequence = [(zone['center']['x'] / 10000, zone['center']['y'] / 10000) for zone in zones[:phase_to_predict - 1]]
                target = (zones[phase_to_predict - 1]['center']['x'] / 10000, zones[phase_to_predict - 1]['center']['y'] / 10000)
                self.data.append((torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def constraint_loss(outputs, targets, blacklist_area, zone_5_radius, blacklist_multiplier=2.0, max_radius_multiplier=.35):
    """
    Loss function with penalty multipliers:
    1. 1.5x loss multiplier if the predicted center is inside the blacklisted area.
    2. Up to 0.5x additional multiplier based on the distance from the ideal Zone 5 radius.
    """
    mse_loss = nn.MSELoss()
    base_loss = mse_loss(outputs, targets[:, :2])  # Standard MSE loss for x and y coordinates

    # Unpack the blacklisted area boundaries
    x_min, y_min, x_max, y_max = blacklist_area

    # Initialize penalty multiplier
    penalty_multiplier = 1.0

    # Apply 1.5x multiplier if the predicted center falls within the blacklisted area
    for i in range(outputs.size(0)):
        x, y = outputs[i, 0], outputs[i, 1]
        if x_min < x < x_max and y_min < y < y_max:
            penalty_multiplier *= blacklist_multiplier  # Apply 1.5x multiplier

    # For Zone 5: Calculate distance from the predicted center to the ideal radius
    predicted_center = outputs[:, :2]
    actual_center = targets[:, :2]  # Zone 5 center
    distance_from_center = torch.sqrt((predicted_center[:, 0] - actual_center[:, 0])**2 +
                                      (predicted_center[:, 1] - actual_center[:, 1])**2)

    # Calculate the radius difference and apply up to 0.5x multiplier based on deviation from zone 5 radius
    radius_diff = torch.abs(distance_from_center - zone_5_radius)
    radius_multiplier = torch.clamp(radius_diff / zone_5_radius, min=0, max=1) * max_radius_multiplier

    # Final penalty multiplier is applied to the base loss
    penalty_multiplier += radius_multiplier.mean().item()  # Add the average radius multiplier

    total_loss = base_loss * penalty_multiplier
    return total_loss

def train_and_save_model(file_path, phase_to_predict, num_epochs=250, learning_rate=0.0025, model_save_path='zone_predictor.pth'):
    """
    Trains the model using the constraint-aware loss function and saves the model after training.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    dataset = ZoneDataset(data, phase_to_predict)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_zone_predictor_for_phase(phase_to_predict).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define the blacklisted area and zone 5 radius for constraints
    blacklist_area = (-120687, -72489, -71884, 36768)
    zone_5_radius = 2000  # Replace with the actual radius for zone 5 in your dataset

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the constraint-aware loss
            loss = constraint_loss(outputs, targets, blacklist_area, zone_5_radius)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print the loss for every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == '__main__':
    file_path = './data/extracted_zone_data.json'
    phase_to_predict = 6
    train_and_save_model(file_path, phase_to_predict)
