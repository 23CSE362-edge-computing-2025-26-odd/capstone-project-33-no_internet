# edgeComputing/client.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from model import CNNModel
from dataset import get_mnist_loaders


class Client:
    # Add data_subset=None to the __init__ parameters
    def __init__(self, client_id, data_subset=None, batch_size=32, lr=0.01, local_epochs=1, device=None):
        self.client_id = client_id
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Use the provided data_subset if it exists
        if data_subset:
            self.train_loader = DataLoader(data_subset, batch_size=batch_size, shuffle=True)
        else:
            # Fallback to loading the full dataset if no subset is provided
            train_loader_full, _ = get_mnist_loaders()
            self.train_loader = train_loader_full

        # The sample_size parameter is no longer needed since data_subset is handled by the simulator.
        # It's better to remove it to avoid confusion and redundant logic.

        # Initialize model
        self.model = CNNModel().to(self.device)

        # Load base weights directly from local file
        self.base_state_dict = torch.load("../src/current_weights.pth", map_location=self.device)
        self.model.load_state_dict(self.base_state_dict)
        print(f"[Client {self.client_id}] Base model loaded from local current_weights.pth.")

    def train(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(
                f"Client {self.client_id} Epoch {epoch + 1}/{self.local_epochs} Loss: {running_loss / len(self.train_loader):.4f}")

    def get_model_update(self):
        """
        Returns the model update (delta) compared to stored base weights
        """
        delta = {}
        current_weights = self.model.state_dict()

        for key in current_weights.keys():
            delta[key] = current_weights[key] - self.base_state_dict[key]

        return delta


# ---------------- Example usage ---------------- #
if __name__ == "__main__":
    # Note: The data_subset parameter is typically provided by the simulator.
    # For standalone testing, you'll need to create a dummy subset.
    print("This script is meant to be run by the simulator. Example usage is for testing only.")
    from torch.utils.data import Dataset


    class DummyDataset(Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item()


    dummy_client_data = DummyDataset()
    client0 = Client(client_id=0, data_subset=dummy_client_data, local_epochs=1)
    client0.train()
    update = client0.get_model_update()
    print("Client 0 update keys:", list(update.keys())[:5])