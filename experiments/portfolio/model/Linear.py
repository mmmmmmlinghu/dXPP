import torch
import torch.nn as nn
from model.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_assets = configs.num_channels
        self.horizon = configs.output_dim
        self.revin = configs.revin
        
        if self.revin:
            self.rev = RevIN(self.num_assets)
        
        # Directly map from asset dimension N to H * N.
        # This makes the model independent of the input sequence length L.
        self.linear = nn.Linear(self.num_assets, self.horizon * self.num_assets)

    def forward(self, x):
        # x: [B, L, N]
        B, L, N = x.shape
        
        if self.revin:
            x = self.rev(x, 'norm')
        
        # Use only the last time step for prediction [B, N]
        last_step = x[:, -1, :]
        
        # Map to [B, H * N]
        pred_flat = self.linear(last_step)
        
        # Reshape back to [B, H, N]
        pred = pred_flat.reshape(B, self.horizon, N)

        if self.revin:
            pred = self.rev(pred, 'denorm')
            
        return pred
