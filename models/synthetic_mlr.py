# models/synthetic_mlr.py
#from models.mlr import MLR

#class SyntheticMLR(MLR):
 #   def __init__(self):
  #      super().__init__(input_dim=60)  # Override input dimension for synthetic data (60D instead of 784)


import torch.nn as nn
import torch.nn.functional as F

class SyntheticMLR(nn.Module):
    def __init__(self, input_dim=60, output_dim=10):
        super(SyntheticMLR, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)
    
    def get_logits(self, x):
        return self.fc(x)