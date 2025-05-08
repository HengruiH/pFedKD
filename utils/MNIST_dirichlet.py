import torch
import numpy as np
from torchvision import datasets, transforms
import os

def load_mnist_dirichlet(n_clients=20, alpha_diric=0.5):
    """
    Partition MNIST data across clients using a Dirichlet distribution to control non-IID heterogeneity.
    
    Args:
        n_clients (int): Number of clients (default: 20).
        alpha_diric (float): Dirichlet concentration parameter (default: 0.5). Lower alpha_diric increases non-IID.
    
    Returns:
        List of (train_X, train_y, val_X, val_y) per client, where X is [n, 784], y is labels.
    """
    # Load and normalize MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Define sample sizes per client (1,165 to 3,834, matching load_mnist_clustered)
    data_per_client = np.linspace(1165, 3834, n_clients, dtype=int)
    
    # Initialize client data
    client_data = []
    n_classes = 10  # MNIST digits 0-9
    available_indices = [torch.where(mnist_train.targets == i)[0] for i in range(n_classes)]
    
    # Dirichlet partitioning
    for client_idx in range(n_clients):
        n_samples = data_per_client[client_idx]
        
        # Sample class proportions from Dirichlet
        proportions = np.random.dirichlet([alpha_diric] * n_classes)

        top_k = 3  # Limit to 2 digits
        top_indices = np.argsort(proportions)[-top_k:]
        new_proportions = np.zeros(n_classes)
        new_proportions[top_indices] = proportions[top_indices] / proportions[top_indices].sum()
        proportions = new_proportions
        
        # Compute samples per class, ensuring sum equals n_samples
        class_counts = np.floor(proportions * n_samples).astype(int)
        class_counts[-1] += n_samples - class_counts.sum()  # Adjust last class for rounding
        
        # Collect samples
        X, y = [], []
        for class_idx in range(n_classes):
            if class_counts[class_idx] == 0:
                continue
            # Randomly sample from available indices
            indices = available_indices[class_idx]
            if len(indices) < class_counts[class_idx]:
                raise ValueError(f"Not enough samples for class {class_idx}: {len(indices)} < {class_counts[class_idx]}")
            selected = indices[torch.randperm(len(indices))[:class_counts[class_idx]]]
            X.append(torch.stack([mnist_train[i][0] for i in selected]))
            y.append(mnist_train.targets[selected])
            
        
        # Concatenate and shuffle
        X = torch.cat(X).view(-1, 784)
        y = torch.cat(y)
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        # Split into train (75%) and validation (25%)
        split = int(0.75 * len(X))
        client_data.append((X[:split], y[:split], X[split:], y[split:]))
    
    return client_data



# Ensure data directory exists
os.makedirs("data", exist_ok=True)
    
# Save to a single .pt file
client_data = load_mnist_dirichlet(n_clients=20, alpha_diric=0.05)
torch.save(client_data, "data/MNIST_N20_a005.pt")

client_data = load_mnist_dirichlet(n_clients=20, alpha_diric=0.5)
torch.save(client_data, "data/MNIST_N20_a05.pt")

client_data = load_mnist_dirichlet(n_clients=50, alpha_diric=0.05)
torch.save(client_data, "data/MNIST_N50_a005.pt")

client_data = load_mnist_dirichlet(n_clients=50, alpha_diric=0.5)
torch.save(client_data, "data/MNIST_N50_a05.pt")



print_proportions_tables(client_data)

def print_proportions_tables(client_data, n_classes=10):
    """
    Print two tables showing class proportions (%) for training and validation sets,
    with clients 1-20 as rows and digits 0-9 as columns.
    
    Args:
        client_data: List of (train_X, train_y, val_X, val_y) from load_mnist_dirichlet.
        n_classes (int): Number of classes (default: 10 for MNIST digits 0-9).
    
    Prints:
        Training table and validation table with proportions (%).
    """
    # Compute proportions
    train_props = []
    val_props = []
    for train_X, train_y, val_X, val_y in client_data:
        train_counts = torch.bincount(train_y, minlength=n_classes)
        val_counts = torch.bincount(val_y, minlength=n_classes)
        train_props.append(train_counts.float() / len(train_y) * 100)
        val_props.append(val_counts.float() / len(val_y) * 100)
    
    # Training table
    print("\nTraining Proportions (%)")
    print("Client   |" + "|".join([f"   {i}   " for i in range(n_classes)]) + "|")
    print("-" * 9 + "|" + "|".join(["-" * 7 for _ in range(n_classes)]) + "|")
    for i, props in enumerate(train_props, 1):
        print(f"Client {i:<2} |" + "|".join([f"{p:>7.2f}" for p in props]) + "|")
    
    # Validation table
    print("\nValidation Proportions (%)")
    print("Client   |" + "|".join([f"   {i}   " for i in range(n_classes)]) + "|")
    print("-" * 9 + "|" + "|".join(["-" * 7 for _ in range(n_classes)]) + "|")
    for i, props in enumerate(val_props, 1):
        print(f"Client {i:<2} |" + "|".join([f"{p:>7.2f}" for p in props]) + "|")

