#pfedme.py
import torch
import random
import torch.nn.functional as F
from utils.train_utils import evaluate
import copy

class UserpFedMe:
    def __init__(self, user_id, data, model, device, local_epochs, batch_size, learning_rate, lamda, K, personal_learning_rate):
        self.id = user_id
        self.X_train, self.y_train, self.X_test, self.y_test = [d.to(device) for d in data]
        self.model = model  # theta_i, updated to theta_i^K
        self.local_model = copy.deepcopy(model)  # w^t, updated to w_i^{t+1}
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=personal_learning_rate)
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate  # eta
        self.lamda = lamda
        self.K = K
    
    def set_parameters(self, global_model):
        """Set local_model to global parameters."""
        self.local_model.load_state_dict(global_model.state_dict())
    
    def train(self):
        """Train as per original pFedMe."""
        self.model.train()
        n_samples = self.X_train.size(0)
        
        for _ in range(self.local_epochs):
            X, y = self.get_next_train_batch(n_samples)
            for _ in range(self.K):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = F.nll_loss(output, y)
                loss.backward()
                # Manually apply Moreau envelope term: lambda * (theta_i^k - w^t)
                for param, l_param in zip(self.model.parameters(), self.local_model.parameters()):
                    param.grad.data += self.lamda * (param.data - l_param.data)
                self.optimizer.step()
        
        # Update local_model: w_i^{t+1} = w^t - lambda * eta * (w^t - theta_i^K)
        local_params = self.local_model.state_dict()
        updated_params = self.model.state_dict()
        for key in local_params:
            local_params[key] -= self.lamda * self.learning_rate * (local_params[key] - updated_params[key])
        self.local_model.load_state_dict(local_params)
        return self.local_model.state_dict()
    
    def get_next_train_batch(self, n_samples):
        """Sample a random batch from training data."""
        indices = torch.randperm(n_samples)[:self.batch_size]
        return self.X_train[indices], self.y_train[indices]
    
    def evaluate(self):
        """Evaluate global model (self.model after set_parameters)."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X_train)
            loss = F.nll_loss(output, self.y_train).item() # Loss on training data
            accuracy = evaluate(self.model, self.X_test, self.y_test, self.device, self.batch_size)
        return accuracy, loss
    
    def evaluate_personalized(self):
        """Evaluate personalized model (self.model after training)."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X_test)
            loss = F.nll_loss(output, self.y_test).item()
            accuracy = evaluate(self.model, self.X_test, self.y_test, self.device, self.batch_size)
        return accuracy, loss

class pfedmeServer:
    def __init__(self, client_data, model_class, device, local_epochs, batch_size, learning_rate,
                 b, lamda, c, K, personal_learning_rate):
        self.users = [
            UserpFedMe(i, client_data[i], model_class().to(device), device, local_epochs, batch_size,
                       learning_rate, lamda, K, personal_learning_rate)
            for i in range(len(client_data))
        ]
        self.global_model = model_class().to(device)
        self.c = c  # Fraction of users to aggregate
        self.beta = b
        self.device = device
    
    def send_parameters(self):
        """Send global model parameters to all users."""
        for user in self.users:
            user.set_parameters(self.global_model)
    
    def train(self, rounds):
        accuracies = []
        losses = []
        for r in range(rounds):
            print(f"pFedMe Round {r+1}/{rounds}")
            self.send_parameters()
            
            # Train all users
            local_params = [user.train() for user in self.users]
            
            # Select subset for aggregation
            num_selected_users = max(1, int(self.c * len(self.users)))
            selected_users = random.sample(self.users, num_selected_users)
            selected_indices = [user.id for user in selected_users]
            
            # Aggregate selected users' local models
            #total_train = sum(len(self.users[i].X_train) for i in selected_indices)
            global_dict = self.global_model.state_dict()
            for key in global_dict:
                global_dict[key] = torch.zeros_like(global_dict[key])
                for i in selected_indices:
                    global_dict[key] += local_params[i][key] * (1 / num_selected_users)
            
            # Beta-weighted update
            previous_params = copy.deepcopy(self.global_model.state_dict())
            for key in global_dict:
                global_dict[key] = (1 - self.beta) * previous_params[key] + self.beta * global_dict[key]
            self.global_model.load_state_dict(global_dict)
            
            # Evaluate personalized models
            avg_accuracy, avg_loss = self.evaluate_personalized()
            accuracies.append(avg_accuracy)
            losses.append(avg_loss)
            print(f"Avg Personalized Accuracy = {avg_accuracy:.2f}%, Avg Loss = {avg_loss:.4f}")
        
        return accuracies, losses
    
    def evaluate(self):
        """Evaluate global model on all users."""
        for user in self.users:
            user.model.load_state_dict(self.global_model.state_dict())
        user_results = [user.evaluate() for user in self.users]
        accuracies, losses = zip(*user_results)
        return sum(accuracies) / len(self.users) * 100, sum(losses) / len(self.users)
    
    def evaluate_personalized(self):
        """Evaluate personalized models on all users."""
        user_results = [user.evaluate_personalized() for user in self.users]
        accuracies, losses = zip(*user_results)
        return sum(accuracies) / len(self.users) * 100, sum(losses) / len(self.users)