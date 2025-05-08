# algorithms/pfedkd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils.train_utils import evaluate

class pfedkdUser:
    def __init__(self, user_id, data, model, device, local_epochs, batch_size, learning_rate, kd_weight):
        self.id = user_id
        self.X_train, self.y_train, self.X_test, self.y_test = [d.to(device) for d in data]
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.kd_weight = kd_weight
    
    def train(self, global_model):
        self.model.train()
        for _ in range(self.local_epochs):
            indices = torch.randperm(len(self.X_train))[:self.batch_size].to(self.device)
            X, y = self.X_train[indices], self.y_train[indices]
            self.optimizer.zero_grad()
            
            output = self.model(X)
            ce_loss = F.nll_loss(output, y)
            
            with torch.no_grad():
                global_probs = F.softmax(global_model(X), dim=1)
            local_probs = F.softmax(self.model(X), dim=1)
            kd_loss = F.kl_div(local_probs.log(), global_probs, reduction='batchmean')
            
            loss = (1 - self.kd_weight) * ce_loss + self.kd_weight * kd_loss
            loss.backward()
            self.optimizer.step()
        
        return self.model
    
    def compute_kl_gradients(self, global_model):
        """Compute gradients of KL(P_l || P_g) w.r.t. global model parameters."""
        global_model.zero_grad() # Reset gradients of global model
        self.model.eval() # Local model is fixed
        global_model.train() # Global model is being updated
        X = self.X_train # Use entire local training data
        with torch.no_grad():
            local_probs = F.softmax(self.model(X), dim=1)
        global_probs = F.softmax(global_model(X), dim=1)
        kl_loss = F.kl_div(local_probs.log(), global_probs, reduction='batchmean')  # Corrected direction
        kl_loss.backward()
        return [param.grad.clone() for param in global_model.parameters()]
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X_train)
            loss = F.nll_loss(output, self.y_train).item() # Loss on training data
            accuracy = evaluate(self.model, self.X_test, self.y_test, self.device, self.batch_size)
        return accuracy, loss

class pfedkdServer:
    def __init__(self, client_data, model_class, device, local_epochs, batch_size, learning_rate, kd_weight, c):
        self.users = [
            pfedkdUser(
                i, 
                client_data[i], 
                model_class().to(device), 
                device, 
                local_epochs, 
                batch_size, 
                learning_rate, 
                kd_weight
            ) 
            for i in range(len(client_data))
        ]
        self.global_model = model_class().to(device)
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=learning_rate)
        self.device = device
        self.c = c
    
    def train(self, rounds):
        accuracies = []
        losses = []
        for r in range(rounds):
            self.aggregate()
            avg_accuracy, avg_loss = self.evaluate()
            accuracies.append(avg_accuracy)
            losses.append(avg_loss)
            print(f"pFedKD Round {r+1}: Avg Personalized Accuracy = {avg_accuracy:.2f}%, Avg Personalized Loss = {avg_loss:.4f}")
        return accuracies, losses
    
    def aggregate(self):
        # Randomly select c * len(users) users
        num_selected_users = max(1, int(self.c * len(self.users)))  # Ensure at least 1 user
        selected_users = random.sample(self.users, num_selected_users)
        
        # Train selected users and collect their models
        #local_models = [user.train(self.global_model) for user in selected_users]
        
        # Aggregate gradients from selected users
        total_train = sum(len(user.X_train) for user in selected_users)
        self.optimizer.zero_grad()
        for user in selected_users:
            user.train(self.global_model)
            gradients = user.compute_kl_gradients(self.global_model)
            weight = len(user.X_train) / total_train
            for g_param, grad in zip(self.global_model.parameters(), gradients):
                if g_param.grad is None:
                    g_param.grad = torch.zeros_like(g_param)
                g_param.grad.add_(grad * weight)
        self.optimizer.step()
    
    def evaluate(self):
        # Evaluate personalized models (local models), not global model
        user_results = [user.evaluate() for user in self.users]
        accuracies, losses = zip(*user_results)
        return sum(accuracies) / len(self.users) * 100, sum(losses) / len(self.users)