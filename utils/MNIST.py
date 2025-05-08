
import torch
import numpy as np
from torchvision import datasets, transforms

def load_mnist_clustered(n_clients=20, n_clusters=20):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    data_per_client = np.linspace(1165, 3834, n_clients, dtype=int)
    labels = torch.arange(10)
    
    clients_per_cluster = n_clients // n_clusters
    client_data = []
    for cluster_idx in range(n_clusters):
        label_pair = labels[torch.randperm(10)[:2]]
        for _ in range(clients_per_cluster):
            n_samples = data_per_client[len(client_data)]
            mask = (mnist_train.targets == label_pair[0]) | (mnist_train.targets == label_pair[1])
            indices = torch.where(mask)[0][torch.randperm(mask.sum())[:n_samples]]
            X = torch.stack([mnist_train[i][0] for i in indices]).view(-1, 784)
            y = mnist_train.targets[indices]
            split = int(0.75 * len(X))
            client_data.append((X[:split], y[:split], X[split:], y[split:]))
    return client_data

client_data = load_mnist_clustered(n_clients=50, n_clusters=10)
torch.save(client_data, "data/MNIST_N50_a05.pt")
# Iterate and print details for each client
for i, (X_train, y_train, X_test, y_test) in enumerate(client_data):
    print(f"Client {i+1}:Train Sample Size: {len(X_train)}, Test Sample Size: {len(X_test)}, Unique Labels: {torch.unique(y_train).tolist()}\n")
   
