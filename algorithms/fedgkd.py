# algorithms/fedgkd.py
import torch
import random
import torch.nn.functional as F
from utils.train_utils import evaluate
import copy
from utils.train_utils import move_to_device

class FedGKDUser:
    def __init__(self, user_id, data, model, device, local_epochs, batch_size, learning_rate, temperature, alpha, buffer_length):
        self.id = user_id
        self.X_train, self.y_train, self.X_test, self.y_test = [move_to_device(d, device) for d in data]
        self.model = model
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.temperature = temperature  # Distillation temperature
        self.alpha = alpha  # Weight for distillation loss vs. CE loss
        self.buffer_length = buffer_length  # Number of historical models for teacher
        self.teacher_model = copy.deepcopy(model).to(device)
        self.teacher_model.eval()  # Teacher is in eval mode
        self.is_bert = hasattr(model, 'distilbert')
        # Choose optimizer based on model type
        if self.is_bert:
            # AdamW for DistilBERT / Transformer models
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.0   
            )
        else:
            # SGD for all other models
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate
            )

    def train(self, global_model, ensembled_teacher):
        # Load global model weights
        self.model.load_state_dict(global_model.state_dict())
        # Update teacher with ensembled teacher weights
        self.teacher_model.load_state_dict(ensembled_teacher.state_dict())
        self.model.train()
        for _ in range(self.local_epochs):
            indices = torch.randperm(len(self.y_train)).to(self.device)
            for i in range(0, len(self.y_train), self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, len(self.y_train))]
                
                if self.is_bert:  # BERT-style inputs
                    input_ids, attention_mask = self.X_train
                    X_batch = (input_ids[batch_indices], attention_mask[batch_indices])
                else:  # CNN/MNIST-style inputs
                    X_batch = self.X_train[batch_indices]

                y_batch = self.y_train[batch_indices]

                self.optimizer.zero_grad()
                # Local model (student) logits
                local_output = self.model(X_batch)
                # Cross-entropy loss
                ce_loss = F.nll_loss(local_output, y_batch)
                local_logits = self.model.get_logits(X_batch)
                # Distillation loss (KL divergence)
                with torch.no_grad():
                    teacher_logits = self.teacher_model.get_logits(X_batch)
                local_soft = F.log_softmax(local_logits / self.temperature, dim=1)
                teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
                kd_loss = F.kl_div(local_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
                # Combined loss
                loss = self.alpha/2 * kd_loss + ce_loss

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

class FedGKDServer:
    def __init__(self, client_data, model_class, device, local_epochs, batch_size, learning_rate, temperature, alpha, buffer_length, c):
        self.users = [
            FedGKDUser(
                i, client_data[i], model_class().to(device), device,
                local_epochs, batch_size, learning_rate, temperature, alpha, buffer_length
            )
            for i in range(len(client_data))
        ]
        self.global_model = model_class().to(device)
        self.ensembled_teacher = copy.deepcopy(self.global_model)
        self.models_buffer = []  # Store historical global models
        self.buffer_length = buffer_length
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
            print(f"FedGKD Round {r+1}: Avg Accuracy = {avg_accuracy:.2f}%, Avg Loss = {avg_loss:.4f}")
        return accuracies, losses

    def aggregate(self):
        # Select clients
        num_selected_users = max(1, int(self.c * len(self.users)))
        selected_users = random.sample(self.users, num_selected_users)
        # Train local models
        local_params = [user.train(self.global_model, self.ensembled_teacher) for user in selected_users]
        total_train = sum(len(user.y_train) for user in selected_users)
        # Aggregate local models
        global_dict = self.global_model.state_dict()
        for key in global_dict:
            global_dict[key] = torch.zeros_like(global_dict[key])
            for user, params in zip(selected_users, local_params):
                global_dict[key] += params[key] * (len(user.y_train) / total_train)
        self.global_model.load_state_dict(global_dict)
        # Update ensembled teacher with historical models
        self._ensemble_historical_models()

    def _ensemble_historical_models(self):
        # Add current global model to buffer
        if len(self.models_buffer) >= self.buffer_length:
            self.models_buffer.pop(0)
        self.models_buffer.append(copy.deepcopy(self.global_model.state_dict()))
        # Average historical models
        if self.models_buffer:
            avg_state_dict = {}
            avg_weight = 1.0 / len(self.models_buffer)
            for key in self.global_model.state_dict():
                avg_state_dict[key] = torch.zeros_like(self.global_model.state_dict()[key])
                for model_dict in self.models_buffer:
                    avg_state_dict[key] += model_dict[key] * avg_weight
            self.ensembled_teacher.load_state_dict(avg_state_dict)

    def evaluate(self):
        for user in self.users:
            user.model.load_state_dict(self.global_model.state_dict())
        user_results = [user.evaluate() for user in self.users]
        accuracies, losses = zip(*user_results)
        return sum(accuracies) / len(self.users) * 100, sum(losses) / len(self.users)