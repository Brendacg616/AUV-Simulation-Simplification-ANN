# MODEL STRUCTURE

import torch
import torch.nn as nn
import torch.nn.functional as F

class MotorControlModelv2(nn.Module):
    def __init__(self, input_size=3, output_size=1, n_neurons=32):  
        super(MotorControlModelv2, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, n_neurons) 
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.output = nn.Linear(n_neurons, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, errors):
        dx = F.relu(self.fc1(errors))
        dx = self.dropout(dx)
        dx = F.relu(self.fc2(dx))
        dx = self.output(dx)
        return dx