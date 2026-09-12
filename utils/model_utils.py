import torch
import torch.nn as nn
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNResidualBlock, self).__init__()

        # calculate padding for causal convolution
        padding = (kernel_size - 1) * dilation if padding == 'causal' else padding 

        # first convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=1, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)

        #shortcut connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ArcNN(nn.Module):
    def __init__(self, cfgs):
        super(ArcNN, self).__init__()
        num_inputs = cfgs[cfgs['model']]['num_inputs']
        num_classes = cfgs['num_classes']
        self.init_convs = nn.Conv1d(in_channels=num_inputs, out_channels=2, kernel_size=1, padding=0)

        self.tcn1 = TCNResidualBlock(in_channels=2, out_channels=10, kernel_size=40, stride=1, dilation=1, padding='causal', dropout=0.2)
        self.tcn2 = TCNResidualBlock(in_channels=10, out_channels=2, kernel_size=40, stride=1, dilation=2, padding='causal', dropout=0.2)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, extract_features=False):
        # SỬA LỖI TIỀM ẨN Ở ĐÂY: Sử dụng unsqueeze để đảm bảo kích thước 3D (Batch, Channels, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  
            
        x = self.init_convs(x)
        x = self.tcn1(x)
        x = self.tcn2(x)
        features = self.flatten(x)
        
        if extract_features:
            return features
        output = self.fc(features)
        return output
    
class SHLNN(nn.Module):
    def __init__(self, cfgs):
        super(SHLNN, self).__init__()

        num_inputs = cfgs[cfgs['model']]['num_inputs']
        num_classes = cfgs['num_classes']
        self.model = nn.Sequential(
            nn.Linear(num_inputs, 110),
            nn.ReLU(True),
            nn.Linear(110, num_classes),
        )

    def forward(self, x):
        return self.model(x)