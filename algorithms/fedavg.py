# algorithms/fedavg.py
import torch
import random
import torch.nn.functional as F
from utils.train_utils import evaluate
from utils.train_utils import move_to_device

class FedAvgUser:
    def __init__(self, user_id, data, model, device, local_epochs, batch_size, learning_rate):
        self.id = user_id
        self.X_train, self.y_train, self.X_test, self.y_test = [move_to_device(d, device) for d in data]
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.is_bert = hasattr(model, 'distilbert')
    
    def train(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        for _ in range(self.local_epochs):
            indices = torch.randperm(len(self.y_train)).to(self.device)        #1 change to y_train
            for i in range(0, len(self.y_train), self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, len(self.y_train))]
                
                if self.is_bert:  # BERT-style inputs
                    input_ids, attention_mask = self.X_train
                    X_batch = (input_ids[batch_indices], attention_mask[batch_indices])
                else:  # CNN/MNIST-style inputs
                    X_batch = self.X_train[batch_indices]

                y_batch = self.y_train[batch_indices]
             
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                ce_loss = F.nll_loss(output, y_batch)
                loss = ce_loss 
                loss.backward()
                self.optimizer.step()
        return self.model.state_dict()
    
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

class FedAvgServer:
    def __init__(self, client_data, model_class, device, local_epochs, batch_size, learning_rate, c):
        self.users = [
            FedAvgUser(i, client_data[i], model_class().to(device), device, local_epochs, batch_size, learning_rate)
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
            print(f"FedAvg Round {r+1}: Avg Accuracy = {avg_accuracy:.2f}%, Avg Loss = {avg_loss:.4f}")
        return accuracies, losses
    
    def aggregate(self):
        num_selected_users = max(1, int(self.c * len(self.users)))
        selected_users = random.sample(self.users, num_selected_users)
        local_params = [user.train(self.global_model) for user in selected_users]
        total_train = sum(len(user.y_train) for user in selected_users)
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
            for user, params in zip(selected_users, local_params):
                global_dict[key] += params[key] * (len(user.y_train) / total_train)  # Correct weighting
        self.global_model.load_state_dict(global_dict)
    
    def evaluate(self):
        for user in self.users:
            user.model.load_state_dict(self.global_model.state_dict())
        user_results = [user.evaluate() for user in self.users]
        accuracies, losses = zip(*user_results)
        return sum(accuracies) / len(self.users) * 100, sum(losses) / len(self.users)