# In a new Python file, e.g., generate_client_updates.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Define the CNNModel class to ensure model compatibility
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_client_updates(client_ids: list, base_model_path: str, output_folder: str = "client_updates"):
    """
    Creates and saves a valid .pth file for each client ID.
    Each file will be a slightly perturbed version of the base model.
    """
    if not client_ids:
        print("No client IDs provided. Please check your editor_log.txt.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the base model (e.g., the global model) to create updates from
    base_model = CNNModel()
    if os.path.exists(base_model_path):
        base_state_dict = torch.load(base_model_path, weights_only=False)
        base_model.load_state_dict(base_state_dict)
    else:
        print(f"Warning: Base model not found at {base_model_path}. Using randomly initialized weights.")

    for client_id in client_ids:
        client_model = CNNModel()
        client_model.load_state_dict(base_model.state_dict())

        # Simulate local training by adding small random noise to the weights
        for param in client_model.parameters():
            if param.requires_grad:
                # Add a small amount of random noise to the parameters
                with torch.no_grad():
                    param.add_(torch.randn(param.size()) * 0.001)

        # Define the path for the client's file
        file_path = os.path.join(output_folder, f"{client_id}.pth")

        # Save the perturbed state dictionary to the file
        torch.save(client_model.state_dict(), file_path)
        print(f"Successfully created update file for client '{client_id}' at {file_path}")

if __name__ == "__main__":
    # --- Configuration ---
    # Define the path to your log file to get the list of clients
    log_file_path = "../edgeComputing/editor_log.txt"
    # Define the path to the base model (e.g., the current global model)
    base_model_path = "./current_weights.pth"
    # The folder where the new client updates will be saved
    client_updates_folder = "../client_updates"

    # Read the client IDs from the log file
    client_ids = []
    try:
        with open(log_file_path, "r") as f:
            client_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")

    # Create the update files
    if client_ids:
        create_client_updates(client_ids, base_model_path, client_updates_folder)