# Description: Multi-layer perceptron model for MNIST dataset.
import torch.nn as nn
import torch.nn.functional as F
import torch
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
    
    def get_logits(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)