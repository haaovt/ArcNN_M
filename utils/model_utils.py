"""
model_utils.py
--------------
ArcNN và SHLNN theo pipeline của paper.

Paper:
Raw current
    -> CBF 90-110 kHz
    -> ArcNN
    -> PCA (80% EVR)
    -> SHLNN

ArcNN:
- input: 512 điểm, 1 channel
- extension channel layer: 1 -> 2
- TCN residual block 1: 2 -> 10, kernel=40, dilation=1
- TCN residual block 2: 10 -> 2, kernel=40, dilation=2
- flatten: 2 x 512 = 1024
"""

import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Cắt phần padding ở cuối để giữ causal convolution."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TCNResidualBlock(nn.Module):
    """
    Residual TCN block.

    Cấu trúc:
        Conv1d
        -> Chomp
        -> BatchNorm
        -> ReLU
        -> Dropout
        -> Conv1d
        -> Chomp
        -> BatchNorm
        -> ReLU
        -> Dropout
        -> residual add
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding="causal",
        dropout=0.2,
    ):
        super().__init__()

        if padding == "causal":
            pad = (kernel_size - 1) * dilation
        else:
            pad = int(padding)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=pad,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.bn1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.bn2,
            self.relu2,
            self.dropout2,
        )

        # Projection shortcut khi số channel thay đổi.
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class ArcNN(nn.Module):
    """
    ArcNN feature extractor/classifier.

    Input:
        (B, 512)
    hoặc:
        (B, 1, 512)

    Output khi extract_features=True:
        (B, 1024)

    Output bình thường:
        (B, 2)

    Paper là bài toán binary classification nên ArcNN dùng 2 classes.
    """

    def __init__(self, cfgs=None):
        super().__init__()

        # Paper: raw waveform có 1 channel.
        self.input_channels = 1

        # Paper: extension channel layer.
        self.init_convs = nn.Conv1d(
            in_channels=1,
            out_channels=2,
            kernel_size=1,
            padding=0,
        )

        # TCN block 1: dilation 1, 10 filters.
        self.tcn1 = TCNResidualBlock(
            in_channels=2,
            out_channels=10,
            kernel_size=40,
            stride=1,
            dilation=1,
            padding="causal",
            dropout=0.2,
        )

        # TCN block 2: dilation 2, 2 filters.
        self.tcn2 = TCNResidualBlock(
            in_channels=10,
            out_channels=2,
            kernel_size=40,
            stride=1,
            dilation=2,
            padding="causal",
            dropout=0.2,
        )

        # 2 channels x 512 time points = 1024 features.
        self.flatten = nn.Flatten()
        self.feature_dim = 1024

        # Binary classification: normal / arc.
        self.fc = nn.Linear(self.feature_dim, 2)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Paper (Sec IV.A): "uniformly initialize filter kernels".
        Dùng kaiming_uniform_ (PyTorch default cho Conv1d với ReLU).
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity='relu',
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, extract_features=False):
        # Dataset có thể trả về (B, 1, 512).
        # Nếu nhận (B, 512), thêm channel dimension.
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.dim() != 3:
            raise ValueError(
                f"ArcNN cần input (B,512) hoặc (B,1,512), "
                f"nhưng nhận shape={tuple(x.shape)}"
            )

        # Bảo vệ khỏi cấu hình num_inputs=5 của config cũ.
        if x.shape[1] != 1:
            raise ValueError(
                f"ArcNN paper dùng 1 input channel, "
                f"nhưng nhận {x.shape[1]} channels."
            )

        x = self.init_convs(x)
        x = self.tcn1(x)
        x = self.tcn2(x)

        features = self.flatten(x)

        # Đây là vector 1024 chiều dùng cho PCA.
        if extract_features:
            return features

        return self.fc(features)


class SHLNN(nn.Module):
    """
    SHLNN classifier sau PCA.

    Paper:
        PCA output: 70 dimensions trên dataset của paper
        70 -> 110 -> 2

    Ở đây num_inputs được truyền từ số chiều PCA thực tế.
    Vì vậy nếu PCA trên dataset hiện tại cho đúng 70 chiều,
    model sẽ tự trở thành 70 -> 110 -> 2.
    """

    def __init__(self, cfgs=None, num_inputs=None):
        super().__init__()

        if num_inputs is None:
            if cfgs is None:
                raise ValueError("Cần num_inputs hoặc cfgs.")
            num_inputs = cfgs.get("shlnn_num_inputs", 70)

        self.num_inputs = int(num_inputs)

        self.model = nn.Sequential(
            nn.Linear(self.num_inputs, 110),
            nn.ReLU(inplace=True),
            nn.Linear(110, 2),
        )

    def forward(self, x):
        return self.model(x)