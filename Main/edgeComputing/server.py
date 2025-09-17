# edgeComputing/server.py


import torch
from model import CNNModel, load_initial_weights
from utils import save_log

class Server:
    def __init__(self, num_clients=3, device=None):
        self.num_clients = num_clients
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize global model
        self.global_model = CNNModel().to(self.device)
        load_initial_weights(self.global_model)
    
    def aggregate_updates(self, client_updates, client_sizes):
        """
        Aggregates updates from clients using FedAvg
        client_updates: list of dicts {layer: tensor delta}
        client_sizes: list of sample counts per client
        """
        total_samples = sum(client_sizes)
        aggregated_update = {}
        
        # Initialize aggregated_update with zeros
        for key in client_updates[0].keys():
            aggregated_update[key] = torch.zeros_like(client_updates[0][key])
        
        # Weighted sum
        for update, size in zip(client_updates, client_sizes):
            for key in update.keys():
                aggregated_update[key] += (size / total_samples) * update[key]
        
        # Apply aggregated update to global model
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] += aggregated_update.get(key, torch.zeros_like(global_state[key]))
        self.global_model.load_state_dict(global_state)
    
    def get_global_weights(self):
        return self.global_model.state_dict()

# Example usage
if __name__ == "__main__":
    from client import Client
    from dataset import get_mnist_loaders
    from partition import non_iid_partition
    from compression import topk_sparsify, should_skip

    # Step 1: prepare clients
    train_loader, _ = get_mnist_loaders()
    train_dataset = train_loader.dataset
    client_subsets = non_iid_partition(train_dataset, num_clients=3)
    
    clients = [Client(i, client_subsets[i], local_epochs=1) for i in range(3)]
    
    # Step 2: local training & updates
    client_updates = []
    client_sizes = []
    for client in clients:
        client.train()
        update = client.get_model_update()
        
        # Simulate bandwidth
        bandwidth = 0.05  # MB
        update_size = 1.0 # MB
        if should_skip(bandwidth, update_size):
            print(f"Client {client.client_id} skipped update due to low bandwidth")
            continue
        
        # Compress update
        sparse_update = topk_sparsify(update, k=0.1)
        client_updates.append(sparse_update)
        client_sizes.append(len(client.train_loader.dataset))
    
    # Step 3: server aggregation
    server = Server(num_clients=3)
    server.aggregate_updates(client_updates, client_sizes)
    print("Global model updated")