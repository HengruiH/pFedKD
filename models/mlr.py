
import torch.nn as nn
import torch.nn.functional as F

class MLR(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MLR, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)
    
    def get_logits(self, x):
        return self.fc(x)