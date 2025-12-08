# MODEL STRUCTURE

import torch
import torch.nn as nn
import torch.nn.functional as F

# Function to stack n layers
import torch
import torch.nn as nn
import torch.nn.functional as F

class thrustFlexNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=3, dropout_enabled=True):
        super(thrustFlexNN, self).__init__()
        self.dropout_enabled = dropout_enabled
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = F.relu(layer(out))
            if self.dropout_enabled:
                out = F.dropout(out, p=0.1)
        out = self.layers[-1](out)
        return out
    
    # NN input size: 