# edgeComputing/client.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from compression import should_skip
from model import CNNModel
from dataset import get_mnist_loaders
from compression import should_skip
import json
import os
client_id = int(os.getenv("CLIENT_ID", 0))




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

    #FL UPDATE
    # --- Flower-compatible wrappers ---

# inside Main/edgeComputing/client.py — replace or add this function

# In client.py

def train_one_round(model_state_dict, epochs=1, client_id=0, results_dir="./results", bw_mbps=None, compression_method="topk"):
    """
    Trains locally and uses a multi-tiered adaptive strategy for sending updates.
    """
    import time, torch
    from model import CNNModel
    from dataset import get_mnist_loaders
    from compression import compress_state_dict
    from instrumentation import write_event, model_bytes_from_state

    # === NEW: Define the bandwidth tiers and compression ratio ranges ===
    # Bandwidth thresholds (in Mbps)
    skip_bw = 0.1
    high_compression_bw_max = 5.0
    med_compression_bw_max = 12.5
    
    # Top-K ratio ranges for compression
    # For Low Bandwidth zone (high compression)
    low_bw_min_ratio = 0.05  # 5%
    low_bw_max_ratio = 0.15  # 15%
    
    # For Medium Bandwidth zone (low compression)
    med_bw_min_ratio = 0.15  # 15%
    med_bw_max_ratio = 0.40  # 40%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Standard Training (no changes here) ---
    model = CNNModel().to(device)
    model.load_state_dict(model_state_dict)
    train_loader, _ = get_mnist_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    train_time_s = time.time() - t0
    
    state_to_send = model.state_dict()
    size_full = model_bytes_from_state(state_to_send)
    write_event(results_dir, f"client{client_id}", "model_size_before_bytes", size_full, "bytes")

    # === NEW MULTI-TIERED DECISION LOGIC ===
    decision = "full"  # Default if bandwidth is high
    payload = state_to_send
    size_after = size_full
    dynamic_topk_ratio = 1.0 # Default to 1.0 (full)

    if bw_mbps is None or bw_mbps < skip_bw:
        # --- Skip Zone ---
        decision = "skip"
        write_event(results_dir, f"client{client_id}", "decision", decision, "")
        return None, len(train_loader.dataset), {"train_time_s": train_time_s, "decision": decision}

    elif bw_mbps < high_compression_bw_max:
        # --- High Compression Zone ---
        decision = "compress"
        zone_width = high_compression_bw_max - skip_bw
        client_pos = bw_mbps - skip_bw
        bw_score = client_pos / zone_width
        
        ratio_range = low_bw_max_ratio - low_bw_min_ratio
        dynamic_topk_ratio = low_bw_min_ratio + (bw_score * ratio_range)

    elif bw_mbps < med_compression_bw_max:
        # --- Medium Compression Zone ---
        decision = "compress"
        zone_width = med_compression_bw_max - high_compression_bw_max
        client_pos = bw_mbps - high_compression_bw_max
        bw_score = client_pos / zone_width
        
        ratio_range = med_bw_max_ratio - med_bw_min_ratio
        dynamic_topk_ratio = med_bw_min_ratio + (bw_score * ratio_range)

    # --- Apply compression if decision was 'compress' ---
    if decision == "compress":
        print(f"[Client {client_id}] Adaptive compression: BW={bw_mbps:.2f} Mbps -> topk_ratio={dynamic_topk_ratio:.2f}")
        compressed_payload = compress_state_dict(state_to_send, method=compression_method, topk_ratio=dynamic_topk_ratio)
        size_after = len(compressed_payload)
        payload = compressed_payload
    
    write_event(results_dir, f"client{client_id}", "decision", decision, "")
    write_event(results_dir, f"client{client_id}", "model_size_after_bytes", size_after, "bytes")

    return model.state_dict(), len(train_loader.dataset), {"train_time_s": train_time_s,
                                                           "decision": decision,
                                                           "model_size_before_bytes": size_full,
                                                           "model_size_after_bytes": size_after}

def evaluate_model(model_state_dict):
    """Evaluate model accuracy on test set."""
    import torch
    from model import CNNModel
    from dataset import get_mnist_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    model.load_state_dict(model_state_dict)
    _, test_loader = get_mnist_loaders()

    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return {"accuracy": correct / total}


#uncompressed function for the client model without compression
# Add this entire new function to the end of your client.py file

def train_one_round_baseline(model_state_dict, client_id=0, results_dir="./results"):
    """
    A simplified training round that ALWAYS sends the full, uncompressed model.
    """
    import time, torch
    from model import CNNModel
    from dataset import get_mnist_loaders
    from instrumentation import write_event, model_bytes_from_state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    model.load_state_dict(model_state_dict)
    train_loader, _ = get_mnist_loaders()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    model.train()
    for _ in range(1): #
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    train_time_s = time.time() - t0
    
    state_to_send = model.state_dict()
    size_full = model_bytes_from_state(state_to_send)
    
    # Log everything as a "full" decision
    write_event(results_dir, f"client{client_id}", "decision", "full", "")
    write_event(results_dir, f"client{client_id}", "model_size_before_bytes", size_full, "bytes")
    write_event(results_dir, f"client{client_id}", "model_size_after_bytes", size_full, "bytes")

    return model.state_dict(), len(train_loader.dataset), {"decision": "full"}