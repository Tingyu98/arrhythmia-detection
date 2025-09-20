import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple 1D CNN for arrhythmia detection.
    Input shape: (batch, 1, input_len)
    Output: class logits (batch, n_classes)
    """
    def __init__(self, input_len: int, n_classes: int = 3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear((input_len // 4) * 64, n_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
