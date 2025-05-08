# Description: Utility functions for training and evaluating models locally.
import torch
import torch.nn.functional as F

def train_local(model, optimizer, X, y, global_model, epochs, batch_size, mu, device):
    """
    Train a model locally for a specified number of epochs with an optional proximal term.
    
    Args:
        model: Model to train.
        optimizer: Optimizer for updating model parameters.
        X: Training data tensor.
        y: Training labels tensor.
        global_model: Global model for proximal term.
        epochs: Number of local epochs.
        batch_size: Size of training batches.
        mu: Proximal term coefficient.
        device: Device to run on.
    
    Returns:
        float: Average loss over all epochs.
    """
    if batch_size > len(X):
        batch_size = len(X)  # Adjust if batch_size exceeds dataset size
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for _ in range(epochs):
        indices = torch.arange(len(X), device=device) #torch.randperm(len(X)).to(device) # Shuffle all indices once per epoch
        for i in range(0, len(X), batch_size):
            batch_indices =  indices[i:i + batch_size]
            X_batch, y_batch = X[batch_indices].to(device), y[batch_indices].to(device)
            optimizer.zero_grad()
            output = model(X_batch)
        
            # Assuming model outputs log-probabilities (e.g., LogSoftmax)
            ce_loss = F.nll_loss(output, y_batch)
        
            if global_model and mu > 0:  # FedProx proximal term
                prox_loss = 0
                for p_local, p_global in zip(model.parameters(), global_model.parameters()):
                    prox_loss += torch.norm(p_local - p_global) ** 2
                loss = ce_loss + (mu / 2) * prox_loss
            else:
                loss = ce_loss
        
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def evaluate(model, X_test, y_test, device, batch_size):
    """
    Evaluate model accuracy on test data in batches.
    
    Args:
        model: Model to evaluate.
        X_test: Test data tensor.
        y_test: Test labels tensor.
        device: Device to run on.
        batch_size: Size of evaluation batches.
    
    Returns:
        float: Accuracy (fraction of correct predictions).
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            X_batch = X_test[i:end_idx].to(device)
            y_batch = y_test[i:end_idx].to(device)
            output = model(X_batch)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(y_batch).sum().item()
            total_samples += len(y_batch)
    
    return total_correct / total_samples if total_samples > 0 else 0.0