# algorithms/perfedavg.py
import torch
import random
import copy
import torch.nn.functional as F
from utils.train_utils import evaluate

class MySGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta=0):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if beta != 0:
                    p.data.add_(-beta, d_p)
                else:
                    p.data.add_(-group['lr'], d_p)
        return loss

class PerFedAvgUser:
    def __init__(self, user_id, data, model, device, local_epochs, batch_size, learning_rate, beta):
        self.id = user_id
        self.X_train, self.y_train, self.X_test, self.y_test = [d.to(device) for d in data]
        self.model = model
        self.optimizer = MySGD(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.beta = beta
    
    def get_next_train_batch(self, data, target):
        indices = torch.randperm(len(data))[:self.batch_size]
        return data[indices], target[indices]
    
    def train(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        for _ in range(self.local_epochs):
            temp_model = copy.deepcopy(list(self.model.parameters()))
            X, y = self.get_next_train_batch(self.X_train, self.y_train)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = F.nll_loss(output, y)
            loss.backward()
            self.optimizer.step()
            
            X, y = self.get_next_train_batch(self.X_train, self.y_train)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = F.nll_loss(output, y)
            loss.backward()
            for old_p, new_p in zip(self.model.parameters(), temp_model):
                old_p.data = new_p.data.clone()
            self.optimizer.step(beta=self.beta)
        return self.model.state_dict()
    
    def train_one_step(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        X, y = self.get_next_train_batch(self.X_train, self.y_train)  # Adapt on training data
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = F.nll_loss(output, y)
        loss.backward()
        self.optimizer.step()
        X, y = self.get_next_train_batch(self.X_train, self.y_train)  # Adapt on training data
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = F.nll_loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.X_train)
            loss = F.nll_loss(output, self.y_train).item() # Loss on training data
            accuracy = evaluate(self.model, self.X_test, self.y_test, self.device, self.batch_size)
        return accuracy, loss

class PerFedAvgServer:
    def __init__(self, client_data, model_class, device, local_epochs, batch_size, learning_rate, beta, c):
        self.users = [
            PerFedAvgUser(i, client_data[i], model_class().to(device), device, local_epochs, batch_size, learning_rate, beta)
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
            avg_accuracy, avg_loss = self.evaluate_one_step()
            accuracies.append(avg_accuracy)
            losses.append(avg_loss)
            print(f"Per-FedAvg Round {r+1}: Avg Personalized Accuracy = {avg_accuracy:.2f}%, Avg Personalized Loss = {avg_loss:.4f}")
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
                global_dict[key] += params[key] * (len(user.X_train) / total_train)
        self.global_model.load_state_dict(global_dict)
    
    def evaluate_one_step(self):
        for user in self.users:
            user.train_one_step(self.global_model)  # Adapt local model from global
        user_results = [user.evaluate() for user in self.users]
        accuracies, losses = zip(*user_results)
        return sum(accuracies) / len(self.users) * 100, sum(losses) / len(self.users)
    
    def evaluate(self): # Optional: kept for global model evaluation if needed
        for user in self.users:
            user.model.load_state_dict(self.global_model.state_dict())
        user_results = [user.evaluate() for user in self.users]
        accuracies, losses = zip(*user_results)
        return sum(accuracies) / len(self.users) * 100, sum(losses) / len(self.users)