import torch
import torch.nn as nn
from torch.nn import functional as F
from model.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin
        self.output_dim = configs.output_dim
        if self.revin:
            self.rev = RevIN(configs.num_channels)
        
        # nn.Sequential requires *modules instead of a list, otherwise it throws TypeError
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=configs.num_channels,
                out_channels=configs.conv_channels,
                kernel_size=configs.conv_kernel_size,
                stride=configs.conv_stride,
                padding='valid',
            ),
            nn.ReLU(),
            nn.Dropout(configs.conv_dropout),
            nn.MaxPool1d(kernel_size=2),
        )

        self.lstm = nn.LSTM(
            input_size=configs.conv_channels,
            hidden_size=configs.lstm_hidden_dim,
            num_layers=configs.lstm_layers,
            batch_first=True,
            dropout=configs.lstm_dropout if configs.lstm_layers > 1 else 0
        )
        
        # Mapping to full h * N dimension via FC layer to ensure prediction steps are not limited by sequence length
        self.fc = nn.Linear(configs.lstm_hidden_dim, configs.output_dim * configs.num_channels)
        
    def forward(self, x):
        B, L, D = x.shape

        if self.revin:
            x = self.rev(x, 'norm')
        
        # 1. Extract local features using CNN
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        # 2. Process temporal information using LSTM, take the last hidden state for global prediction
        _, (h_n, _) = self.lstm(x)
        # h_n: [num_layers, B, hidden_dim]
        last_hidden = h_n[-1] 

        # 3. Output the full prediction window [B, h * N]
        pred = self.fc(last_hidden)
        
        # 4. Reshape back to [B, h, N]
        pred = pred.reshape(B, self.output_dim, -1)

        if self.revin:
            pred = self.rev(pred, 'denorm')
            
        return pred
        