import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import matplotlib.pyplot as plt  # Importing matplotlib for visualization

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


def train_model(model, criterion, optimizer, X_train, y_train, epochs=200, batch_size=32, patience=10):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_values = []  
    val_loss_values = [] 
    best_loss = float('inf')
    epochs_without_improvement = 0


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        average_loss = epoch_loss / len(dataloader)
        loss_values.append(average_loss)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {average_loss}')
        
        scheduler.step()

        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement in the last {patience} epochs.")
            break
    
    # Plotting the loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_values) + 1), loss_values, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Neural Network Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

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

def save_model(model, model_path='data/zone_prediction_model_full.pth'):
    torch.save(model, model_path)  # Save the entire model including architecture
    print(f"Full model saved to {model_path}")

extracted_file = 'data/extracted_zone_data.json'
extracted_data = load_data(extracted_file)
X, y = prepare_data(extracted_data)

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

input_dim = X_train.shape[1]
model = ZonePredictionModel(input_dim) 
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)

trained_model = train_model(model, criterion, optimizer, X_train, y_train, epochs=400, batch_size=128, patience=50)

predictions, y_test = evaluate_model(model, X_test, y_test)

save_model(model, 'data/zone_prediction_model.pth')
scalers = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
with open('data/scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
