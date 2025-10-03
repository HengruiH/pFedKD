# algorithms/pfedkd.py
from matplotlib.pylab import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from utils.train_utils import evaluate
from utils.train_utils import move_to_device

class pfedkdUser:
    def __init__(self, user_id, data, model, device, local_epochs, batch_size, learning_rate, kd_weight):
        self.id = user_id
        self.X_train, self.y_train, self.X_test, self.y_test = [move_to_device(d, device) for d in data]
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.kd_weight = kd_weight
        self.is_bert = hasattr(model, 'distilbert')
    
    def train(self, global_model):
        self.model.train()
        for _ in range(self.local_epochs):
            indices = torch.randperm(len(self.y_train))[:self.batch_size].to(self.device)
            #X, y = self.X_train[indices], self.y_train[indices]
            
            if self.is_bert:  # BERT-style inputs
                    input_ids, attention_mask = self.X_train
                    X = (input_ids[indices], attention_mask[indices])
            else:  # CNN/MNIST-style inputs
                    X = self.X_train[indices]

            y = self.y_train[indices]

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


        indices = torch.arange(len(self.y_train)).to(self.device)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            if self.is_bert:
                input_ids, attention_mask = self.X_train
                X_batch = (input_ids[batch_indices], attention_mask[batch_indices])
            else:
                X_batch = self.X_train[batch_indices]

            with torch.no_grad():
                local_probs = F.softmax(self.model(X_batch), dim=1)

            global_outputs = global_model(X_batch)
            global_probs = F.softmax(global_outputs, dim=1)
        
            kl_loss = F.kl_div(local_probs.log(), global_probs, reduction='batchmean')  # Corrected direction
            kl_loss.backward()

        return [param.grad.clone() for param in global_model.parameters()]
    
    def evaluate(self):
        self.model.eval()
        total_train_loss = 0.0
        total_test_accuracy = 0.0
        num_train_batches = 0
        #num_test_batches = 0
        
        with torch.no_grad():
            # Evaluate on training data in batches
            indices = torch.randperm(len(self.y_train)).to(self.device)
            for i in range(0, len(self.y_train), self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, len(self.y_train))]
                if self.is_bert:  # Handle DistilBERT-style tuple inputs
                    input_ids, attention_mask = self.X_train
                    X_batch = (input_ids[batch_indices], attention_mask[batch_indices])
                else:  # Handle MLP-style single tensor inputs
                    X_batch = self.X_train[batch_indices]
                y_batch = self.y_train[batch_indices]
                
                output = self.model(X_batch)
                train_loss = F.nll_loss(output, y_batch).item()
                total_train_loss += train_loss
                num_train_batches += 1
            
            # Evaluate on test data in batches
            if self.is_bert:
                test_accuracy = evaluate(self.model, self.X_test, self.y_test, self.device, self.batch_size)
            else:
                test_accuracy = evaluate(self.model, self.X_test, self.y_test, self.device, self.batch_size)
            total_test_accuracy = test_accuracy  # evaluate() already processes test data in batches

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        return total_test_accuracy, avg_train_loss

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
        total_train = sum(len(user.y_train) for user in selected_users)
        self.optimizer.zero_grad()
        for user in selected_users:
            user.train(self.global_model)
            gradients = user.compute_kl_gradients(self.global_model)
            weight = len(user.y_train) / total_train
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