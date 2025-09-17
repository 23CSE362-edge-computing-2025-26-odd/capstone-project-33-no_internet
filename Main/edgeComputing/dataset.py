import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=32):
    # Transform: convert to tensor + normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Training dataset
    train_dataset = datasets.MNIST(
        root="data/archive",
        train=True,
        transform=transform,
        download=True
    )

    # Test dataset
    test_dataset = datasets.MNIST(
        root="data/archive",  # <-- same root!
        train=False,
        transform=transform,
        download=True
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage
if __name__ == "__main__":
    train_loader, test_loader = get_mnist_loaders()
    for images, labels in train_loader:
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
        break