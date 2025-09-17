# edgeComputing/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ------------------ Weight helpers (.pth format) ------------------ #

def save_initial_weights(model, filepath="current_weights.pth"):
    """Save model weights to a .pth file"""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"[Model] Weights saved to {filepath}")


def load_initial_weights(model, filepath="current_weights.pth", device=None):
    """Load model weights safely with device mapping"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(filepath):
        state_dict = torch.load(filepath, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Model] Loaded weights from {filepath}")
    else:
        print(f"[Model] No file found at {filepath}, using random initialization")
    return model


# ------------------ Example usage ------------------ #
if __name__ == "__main__":
    model = CNNModel()
    print(model)
    save_initial_weights(model)
    load_initial_weights(model)
