import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os

# Ensure the data directory exists
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Function to load the parsed data
def load_data(extracted_file):
    with open(extracted_file, 'r') as file:
        extracted_data = json.load(file)
    return extracted_data

# Function to prepare the data for training
def prepare_data(extracted_data):
    X = []
    y = []

    for match in extracted_data:
        zones = match['zones']
        if len(zones) >= 5:
            input_data = []
            for i in range(5):
                zone = zones[i]
                input_data.extend([
                    zone['previousCenter']['x'], 
                    zone['previousCenter']['y'], 
                    zone['previousCenter']['z']
                ])
            X.append(input_data)
            next_zone = zones[5]  # Phase 7 data
            y.append([
                next_zone['previousCenter']['x'], 
                next_zone['previousCenter']['y'], 
                next_zone['previousCenter']['z']
            ])
    
    return np.array(X), np.array(y)

# Function to build the neural network model
class ZonePredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(ZonePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Function to train the model
def train_model(model, criterion, optimizer, X_train, y_train, epochs=50, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mae = nn.L1Loss()(predictions, y_test)
        rmse = torch.sqrt(nn.MSELoss()(predictions, y_test))
        print(f"Mean Absolute Error (MAE): {mae.item()}")
        print(f"Root Mean Squared Error (RMSE): {rmse.item()}")
        predictions = predictions.numpy()
        y_test = y_test.numpy()
    return predictions, y_test

# Function to save the model
def save_model(model, model_path='data/zone_prediction_model.pth'):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Load and prepare the data
extracted_file = 'data/extracted_zone_data.json'
extracted_data = load_data(extracted_file)
X, y = prepare_data(extracted_data)

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Build the model
input_dim = X_train.shape[1]
model = ZonePredictionModel(input_dim)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model = train_model(model, criterion, optimizer, X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model
predictions, y_test = evaluate_model(model, X_test, y_test)

# Save the model and scalers
save_model(model, 'data/zone_prediction_model.pth')
scalers = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
with open('data/scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
