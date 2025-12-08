import numpy as np
import joblib  
import torch
# Load motor scalers
#motor_u_scaler = joblib.load(pathname + "/MotorModels/Scalers/u_scaler.pkl")
class torchScaler:
    """A simple scaler class to mimic sklearn's StandardScaler functionality in PyTorch."""
    def __init__(self, scaler_path, device='cpu', dtype=torch.float32):
        scaler = joblib.load(scaler_path)
        self.mean = torch.tensor(scaler.mean_, device=device, dtype=dtype)
        self.scale = torch.tensor(scaler.scale_, device=device, dtype=dtype)

    def transform(self, X):
        return (X - self.mean) / self.scale

    def inverse_transform(self, X_scaled):
        return X_scaled * self.scale + self.mean
    