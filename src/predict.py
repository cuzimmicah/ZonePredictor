import torch
import matplotlib.pyplot as plt
from nn import create_zone_predictor_for_phase

def load_model(model_path, phase_to_predict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_zone_predictor_for_phase(phase_to_predict).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_next_zone(model, zone_centers):
    device = next(model.parameters()).device
    # Normalize input by dividing by 100 to bring the values to meters
    normalized_centers = [(x / 100, y / 100) for x, y in zone_centers]
    input_tensor = torch.tensor(normalized_centers, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()
    # Convert the prediction back to meters directly (no further scaling needed)
    return prediction[0][0] * 100, prediction[0][1] * 100

def plot_zones_and_prediction(zone_centers, predicted_center, radii, actual_center=None):
    x_coords = [zone_centers[-1][0] / 100]  # Only Zone 5 center
    y_coords = [zone_centers[-1][1] / 100]  # Only Zone 5 center
    scaled_radii = [r / 100 for r in radii]  # Scaled radii for all zones

    plt.figure(figsize=(10, 10))
    
    # Plot the center and radius for Zone 5
    plt.scatter(x_coords[0], y_coords[0], color='g', s=100, label='Zone 5 Center')
    plt.gca().add_patch(plt.Circle((x_coords[0], y_coords[0]), scaled_radii[4], color='g', fill=False, linewidth=1.5))
    
    # Plot the radius for Zones 1-4 (without centers)
    for i in range(len(zone_centers) - 1):
        zone_x = zone_centers[i][0] / 100
        zone_y = zone_centers[i][1] / 100
        plt.gca().add_patch(plt.Circle((zone_x, zone_y), scaled_radii[i], color='green', linestyle='solid', fill=False, linewidth=1.2, alpha=0.5))

    # Plot the predicted center
    predicted_x, predicted_y = predicted_center[0], predicted_center[1]
    plt.scatter(predicted_x, predicted_y, color='r', marker='x', s=100, label='Predicted Zone 6 Center')
    plt.gca().add_patch(plt.Circle((predicted_x, predicted_y), scaled_radii[5], color='r', linestyle='--', fill=False, linewidth=1.5))
    
    if actual_center:
        actual_x, actual_y = actual_center[0] / 100, actual_center[1] / 100
        plt.scatter(actual_x, actual_y, color='b', s=100, label='Actual Zone 6 Center')
        plt.gca().add_patch(plt.Circle((actual_x, actual_y), scaled_radii[5], color='b', fill=False, linewidth=1.5))
    
    plt.xlabel('X Coordinate (meters)')
    plt.ylabel('Y Coordinate (meters)')
    plt.title('Zone Prediction')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-1500, 1500)
    plt.ylim(-1500, 1500)
    plt.show()

def main():
    model_path = 'zone_predictor.pth'
    phase_to_predict = 6
    zone_centers = [
        (-16452.34, 15451.76),
        ( -38919.49, 13553.98),
        (-25118.11, 2703.37),
        (-37567.22,  10131.76),
        (-46705.94, 28370.2)
    ]

    radii = [120000, 95000, 70000, 55000, 32500, 20000]

    actual_center = (-79004.12, 24753.94)

    model = load_model(model_path, phase_to_predict)
    predicted_center = predict_next_zone(model, zone_centers)
    print(f'Predicted next zone center: {predicted_center}')

    plot_zones_and_prediction(zone_centers, predicted_center, radii, actual_center)

if __name__ == '__main__':
    main()
