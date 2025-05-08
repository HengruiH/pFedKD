# utils/synthetic.py

import torch
import numpy as np
import os

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic_data(n_clients, alpha=0.5, beta=0.5, iid=False):

    dimension = 60
    num_classes = 10
    
    samples_per_user = (np.random.lognormal(4, 2, (n_clients)).astype(int) + 50) * 5
    #print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(n_clients)]
    y_split = [[] for _ in range(n_clients)]


    # Define priors
    mean_W = np.random.normal(0, alpha, n_clients)
    mean_b = mean_W
    B = np.random.normal(0, beta, n_clients)
    mean_x = np.zeros((n_clients, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(n_clients):
        if iid:
            mean_x[i] = np.ones(dimension) * B[i]
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    if iid:
        W_global = np.random.normal(0, 1, (dimension, num_classes))
        b_global = np.random.normal(0, 1, num_classes)

    for i in range(n_clients):
        W = np.random.normal(mean_W[i], 1, (dimension, num_classes))
        b = np.random.normal(mean_b[i], 1, num_classes)
        if iid:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx  # Keep as NumPy array for now
        y_split[i] = yy

    client_data = []
    for i in range(n_clients):
        X = torch.tensor(X_split[i], dtype=torch.float32)
        y = torch.tensor(y_split[i], dtype=torch.long)

        n_samples = samples_per_user[i]
        train_len = int(0.75 * n_samples)
        indices = torch.randperm(n_samples)
        train_idx, test_idx = indices[:train_len], indices[train_len:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        client_data.append((X_train, y_train, X_test, y_test))

    return client_data    

if __name__ == "__main__":
    # Generate data for 100 clients (matching your experiment)
    n_clients = 100
    data = generate_synthetic_data(n_clients=n_clients, alpha=0.5, beta=0.5, iid=False)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to a single .pt file
    torch.save(data, "data/synthetic_data.pt")
    print(f"Saved synthetic data for {n_clients} clients to data/synthetic_data.pt")


    