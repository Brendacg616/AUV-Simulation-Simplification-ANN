# %%
num_epochs = 10
neurons_per_layer = [64, 128, 256]
hidden_layers = [5, 10, 15]
model_prefix = "scaled_data_model_"

# %%
import numpy as np
import nn_fncs
mat_data = nn_fncs.read_mat_workspace('Thrust_data.mat')
nn_in = mat_data.get('nn_in')
nn_out = mat_data.get('nn_out')
# Get every 5th data sample
nn_in = nn_in[:, ::5, :]
nn_out = nn_out[:, ::5, :]
# remove zero columns in nn_out
nn_out1 = nn_out[:, :, ~np.all(nn_out == 0, axis=(0, 1))]
nn_out2 = nn_out[:, :, [0, 1, 2, 3, 7, 10, 11]] 
# Verify that all elements of nn_out1 and nn_out2 are the same
assert np.array_equal(nn_out1, nn_out2), "The arrays are not equal"
nn_out = nn_out2
print(f'nn_in shape: {nn_in.shape}')  # (num_trajectories, num_time_steps, num_inputs)
print(f'nn_out shape: {nn_out.shape}')      # (num_trajectories, num_time_steps, num_outputs)
# Reshape to 2D arrays for training
num_trajectories, num_time_steps, num_inputs = nn_in.shape
num_outputs = nn_out.shape[2]
nn_in = nn_in.reshape(-1, num_inputs)
nn_out = nn_out.reshape(-1, num_outputs)
print(f'Reshaped nn_in shape: {nn_in.shape}')  # (num_trajectories * num_time_steps, num_inputs)
print(f'Reshaped nn_out shape: {nn_out.shape}')  # (num_

# %%
from sklearn import preprocessing
in_scaled = preprocessing.StandardScaler().fit_transform(nn_in)
print(f'u_scaled shape: {in_scaled.shape}')

# %%
# Check min and max values of nn_in and nn_out per column in the last axis
for i in range(in_scaled.shape[1]):
    print(f'nn_in column {i}: min={np.min(in_scaled[:, i])}, max={np.max(in_scaled[:, i])}')

for i in range(nn_out.shape[1]):
    print(f'nn_out column {i}: min={np.min(nn_out[:, i])}, max={np.max(nn_out[:, i])}')


# %%
# Create data loaders
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Split into training and validation sets (80% train, 20% val)
# separate into training and validation sets
split_ratio = 0.75
split_index = int(in_scaled.shape[0] * split_ratio)
# Shuffle data before splitting
indices = np.arange(in_scaled.shape[0])
np.random.shuffle(indices)
in_scaled = in_scaled[indices]
nn_out = nn_out[indices]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Split data into training and validation sets
train_dataset = torch.utils.data.TensorDataset(torch.tensor(in_scaled[:split_index], dtype=torch.float32).to(device), 
                                               torch.tensor(nn_out[:split_index], dtype=torch.float32).to(device))
valid_dataset = torch.utils.data.TensorDataset(torch.tensor(in_scaled[split_index:], dtype=torch.float32).to(device), 
                                               torch.tensor(nn_out[split_index:], dtype=torch.float32).to(device))

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1000, shuffle=False)

# %%
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
    


# model = ThrustModel(nn_in.shape[1], nn_out.shape[1])

# # Print model summary
# print(model)

# # Count trainable parameters
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f'Total trainable parameters: {total_params}')

# %%
# One step training
import torch.optim as optim
from torcheval.metrics.functional import mean_squared_error, r2_score
from scipy.integrate import odeint

def one_step_training(model, criterion, optimizer, input_data, target, r2_scalar = True):
    t = 0
    loss = 0.0
    xk = target[0]
    predictions = torch.zeros_like(target)
    for i in range(target.shape[0]-1):

        with torch.set_grad_enabled(True):
            # Forward pass
            pred = model(input_data)  # Integrate over a small time step
            loss = criterion(pred, target)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    if r2_scalar:
        r2 = r2_score(pred, target, multioutput='uniform_average')
    else:
        r2 = r2_score(pred, target, multioutput='raw_values')
    return pred, loss, r2

# %%
# One step eval
import torch.optim as optim
from torcheval.metrics.functional import mean_squared_error, r2_score
from scipy.integrate import odeint

def one_eval_step(model, input_data, target, r2_multiout=False):
    loss = 0.0
    model.eval()

    for i in range(target.shape[0]-1):

        with torch.no_grad():
            # Forward pass
            pred = model(input_data)  # Integrate over a small time step
    mse = mean_squared_error(pred, target)
            # Backward and optimize
    if r2_multiout:
        r2 = r2_score(pred, target, multioutput='raw_values')
    else:    
        r2 = r2_score(pred, target)
    return pred, mse, r2

# %%
# Complete training loop
import torch
from tqdm import tqdm

def model_training_loop(model, train_loader, valid_loader, num_epochs=1000, epoch_update=10, model_name='thrust_model'):
    best_r2 = -float('inf')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    history_train = {'loss': [], 'r2': []}
    history_val = {'val_loss': [], 'val_r2': []}
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_r2 = 0.0
            for in_tensor, out_tensor in train_loader:
                pred, loss, r2 = one_step_training(model, 
                                                    criterion, 
                                                    optimizer,
                                                    in_tensor,
                                                    out_tensor)
                epoch_loss += loss.item()*in_tensor.size(0)
                epoch_r2 += r2.item()*in_tensor.size(0)
            epoch_loss /= (train_loader.dataset.tensors[0].shape[0])
            epoch_r2 /= (train_loader.dataset.tensors[1].shape[0])
            history_train['loss'].append(epoch_loss)
            history_train['r2'].append(epoch_r2)
            if (epoch+1) % epoch_update == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6e}, R2: {epoch_r2:.6e}')

            val_loss = 0.0
            val_r2 = 0.0
            for in_tensor, out_tensor in valid_loader:
                pred, loss, r2 = one_eval_step(model, in_tensor, out_tensor)
                val_loss += loss*in_tensor.size(0)
                val_r2 += r2.item()*in_tensor.size(0)
            val_loss /= (valid_loader.dataset.tensors[0].shape[0])
            val_r2 /= (valid_loader.dataset.tensors[1].shape[0])
            if (epoch+1) % epoch_update == 0:
                print(f'Validation Loss: {val_loss:.6e}, Validation R2: {val_r2:.6e}')
                pbar.update(epoch_update)
            if val_r2 > best_r2:
                best_r2 = val_r2
                nn_fncs.save_best_model(model, val_r2, 0, model_name=model_name)
            history_val['val_loss'].append(val_loss)
            history_val['val_r2'].append(val_r2)
    return history_val, history_train, best_r2


# %%
modebest_r2 = -float('inf')
for npl in neurons_per_layer:
    for hl in hidden_layers:
        model_name = model_prefix + f'{npl}_neurons_{hl}_layers_'
        print(f'Training model with {npl} neurons per layer and {hl} hidden layers')
        model = thrustFlexNN(input_size=nn_in.shape[1], hidden_size=npl, output_size=nn_out.shape[1], hidden_layers=hl, dropout_enabled=True)
        history_val, history_train, best_r2 = model_training_loop(model, train_loader, valid_loader, num_epochs=num_epochs, epoch_update=1, model_name=model_name)
        


