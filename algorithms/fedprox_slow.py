# algorithms/fedprox.py
import torch
import random
import torch.nn.functional as F
from utils.train_utils import evaluate

class FedProxUser:
    def __init__(self, user_id, data, model, device, local_epochs, batch_size, learning_rate, mu):
        self.id = user_id
        self.X_train, self.y_train, self.X_test, self.y_test = [d.to(device) for d in data]
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.mu = mu
    
    def train(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        for _ in range(self.local_epochs):
            indices = torch.randperm(len(self.X_train))[:self.batch_size].to(self.device)
            for i in range(0, len(self.X_train), self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, len(self.X_train))]
                X_batch, y_batch = self.X_train[batch_indices], self.y_train[batch_indices]
             
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                ce_loss = F.nll_loss(output, y_batch)
                prox_loss = 0
                for p_local, p_global in zip(self.model.parameters(), global_model.parameters()):
                    prox_loss += torch.norm(p_local - p_global) ** 2
                loss = ce_loss + (self.mu / 2) * prox_loss
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X_train)
            loss = F.nll_loss(output, self.y_train).item() # Loss on training data
            accuracy = evaluate(self.model, self.X_test, self.y_test, self.device, self.batch_size)
        return accuracy, loss

class FedProxServer:
    def __init__(self, client_data, model_class, device, local_epochs, batch_size, learning_rate, mu, c):
        self.users = [
            FedProxUser(i, client_data[i], model_class().to(device), device, local_epochs, batch_size, learning_rate, mu)
            for i in range(len(client_data))
        ]
        self.global_model = model_class().to(device)
        self.c = c
        self.device = device
    
    def train(self, rounds):
        accuracies = []
        losses = []
        for r in range(rounds):
            self.aggregate()
            avg_accuracy, avg_loss = self.evaluate()
            accuracies.append(avg_accuracy)
            losses.append(avg_loss)
            print(f"FedProx Round {r+1}: Avg Accuracy = {avg_accuracy:.2f}%, Avg Loss = {avg_loss:.4f}")
        return accuracies, losses
    
    def aggregate(self):
        num_selected_users = max(1, int(self.c * len(self.users)))
        selected_users = random.sample(self.users, num_selected_users)
        local_params = [user.train(self.global_model) for user in selected_users]
        total_train = sum(len(user.X_train) for user in selected_users)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
            for user, params in zip(selected_users, local_params):
                global_dict[key] += params[key] * (len(user.X_train) / total_train)  # Correct weighting
        self.global_model.load_state_dict(global_dict)
    
    def evaluate(self):
        for user in self.users:
            user.model.load_state_dict(self.global_model.state_dict())
        user_results = [user.evaluate() for user in self.users]
        accuracies, losses = zip(*user_results)
        return sum(accuracies) / len(self.users) * 100, sum(losses) / len(self.users)