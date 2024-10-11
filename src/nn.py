import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json

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

def train_and_save_model(file_path, phase_to_predict, num_epochs=250, learning_rate=0.0025, model_save_path='zone_predictor.pth'):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    dataset = ZoneDataset(data, phase_to_predict)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_zone_predictor_for_phase(phase_to_predict).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')
    
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == '__main__':
    file_path = './data/extracted_zone_data.json'
    phase_to_predict = 6
    train_and_save_model(file_path, phase_to_predict)
