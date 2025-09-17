# edgeComputing/partition.py
import numpy as np
from torch.utils.data import Subset

def non_iid_partition(dataset, num_clients=3):
    """
    Split dataset into non-IID partitions per client.
    Example: Client 0 gets digits 0-2, Client 1 gets 3-5, Client 2 gets 6-9
    """
    targets = np.array(dataset.targets)  # get all labels
    client_indices = []

    classes_per_client = 10 // num_clients
    for i in range(num_clients):
        class_range = list(range(i * classes_per_client, (i+1) * classes_per_client))
        idx = np.where(np.isin(targets, class_range))[0]
        client_indices.append(idx.tolist())
    
    # Convert indices to Subset objects for DataLoader
    client_subsets = [Subset(dataset, idx) for idx in client_indices]
    return client_subsets

# Example usage
if __name__ == "__main__":
    from dataset import get_mnist_loaders
    train_loader, _ = get_mnist_loaders()
    train_dataset = train_loader.dataset
    client_subsets = non_iid_partition(train_dataset, num_clients=3)
    
    for i, subset in enumerate(client_subsets):
        print(f"Client {i} has {len(subset)} samples")