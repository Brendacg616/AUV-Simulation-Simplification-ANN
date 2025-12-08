# MODEL STRUCTURE

import torch
import torch.nn as nn
import torch.nn.functional as F

class TailKinematicsRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=2, num_layers=8, output_size=3, sequence_length=8):  
        super(TailKinematicsRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size*sequence_length+sequence_length+1, output_size)

    def forward(self, x):
        
        out, _ = self.rnn(x[:,:-1,:]) 
        out = self.flatten(out)
        new_x = torch.cat((x.squeeze(-1) , out), dim=1)
        out = self.fc(new_x)  # Get the output of the
        
        return out