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

        # SỬA (wiring bug): trước đây tham số `cfgs` được nhận vào
        # nhưng KHÔNG hề được đọc — dropout luôn hardcode = 0.2 dù
        # sweep_utils.py có random-search cfgs['ArcNN']['dropout'].
        # Hệ quả: mọi lần sweep, dropout thực tế dùng để train luôn là
        # 0.2 bất kể giá trị sweep_utils.py sinh ra -> search vô nghĩa.
        # Giờ đọc dropout từ cfgs['ArcNN']['dropout'] nếu có, fallback
        # về 0.2 (giá trị mặc định cũ) nếu không truyền cfgs.
        arcnn_cfgs = (cfgs or {}).get('ArcNN', {}) if isinstance(cfgs, dict) else {}
        dropout = float(arcnn_cfgs.get('dropout', 0.2))

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
            dropout=dropout,
        )

        # TCN block 2: dilation 2, 2 filters.
        self.tcn2 = TCNResidualBlock(
            in_channels=10,
            out_channels=2,
            kernel_size=40,
            stride=1,
            dilation=2,
            padding="causal",
            dropout=dropout,
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

    Paper (đã đối chiếu với file paper thật, Section III-C):
        PCA output: 70 dimensions trên dataset của paper
        70 -> 110 (ReLU) -> 2 (Sigmoid)

    "The rectified linear unit is used as the nonlinear activation
    function of the hidden layer... Sigmoid function is used as the
    activation function of the output layer to achieve binary
    classification." — output layer 2 neuron dùng SIGMOID, không phải
    raw logits. Nhãn cũng ở dạng one-hot [0,1]/[1,0] (paper: "The final
    output... is the one-hot value... [0,1] and [1,0] for dc arc and
    normal, respectively").

    SỬA (thiếu activation output — phát hiện sau khi đọc paper):
    Bản trước chỉ có `Linear(hidden, 2)` KHÔNG có activation cuối, và
    dùng chung nn.CrossEntropyLoss() với ArcNN ở BaseTrainer — cách
    này ngầm giả định output là logits cho softmax, khác với paper
    (sigmoid, không phải softmax). Đã thêm nn.Sigmoid() vào cuối
    self.model để khớp đúng kiến trúc paper. BaseTrainer (trainer_utils.py)
    đã được cập nhật tương ứng: dùng nn.BCELoss() + target one-hot
    cho SHLNN, giữ nn.CrossEntropyLoss() + target là class-index cho
    ArcNN (paper Section IV-A: "cross-entropy loss" dùng khi train
    ArcNN — không nói rõ lại cho SHLNN, nhưng Sigmoid ở Section III-C
    chỉ áp dụng cho SHLNN).

    XUNG ĐỘT VỚI CONFIG CỦA BẠN — CẦN BẠN QUYẾT ĐỊNH:
    config_stage2.yaml (và cả config.yaml/config_stage1.yaml) đều ghi
    `loss_type: CrossEntropy`, kể cả cho SHLNN — trái với Sigmoid mà
    Section III-C paper mô tả. Trường `loss_type` này HIỆN KHÔNG được
    trainer_utils.py đọc (dead field), nên không rõ đây là quyết định
    có chủ đích hay chỉ copy-paste từ config ArcNN. Vì paper là nguồn
    chính xác hơn, mình để MẶC ĐỊNH theo paper (Sigmoid+BCE). Nếu bạn
    muốn khớp đúng những gì config hiện ghi (CrossEntropy, không
    sigmoid) thay vì paper, đặt `cfgs['SHLNN']['sigmoid_output'] =
    False` (ví dụ thêm `sigmoid_output: false` dưới mục `SHLNN:` trong
    config_stage2.yaml) — code sẽ tự bỏ Sigmoid và trainer_utils.py sẽ
    tự chuyển sang CrossEntropyLoss + target class-index cho SHLNN.

    Ở đây num_inputs được truyền từ số chiều PCA thực tế.
    Vì vậy nếu PCA trên dataset hiện tại cho đúng 70 chiều,
    model sẽ tự trở thành 70 -> hidden -> 2.
    """

    def __init__(self, cfgs=None, num_inputs=None):
        super().__init__()

        cfgs = cfgs or {}
        shlnn_cfgs = cfgs.get('SHLNN', {}) if isinstance(cfgs, dict) else {}

        if num_inputs is None:
            # SỬA: sweep_utils.py đặt số chiều PCA ở
            # cfgs['SHLNN']['num_inputs'] (dict lồng nhau), nhưng bản cũ
            # chỉ đọc cfgs['shlnn_num_inputs'] (key phẳng) -> luôn rơi
            # về giá trị này không tồn tại trong cfgs do sweep_utils.py
            # sinh ra, phải dùng default 70. Giờ ưu tiên đọc từ
            # cfgs['SHLNN']['num_inputs'] trước, fallback về key phẳng
            # cũ để tương thích ngược, rồi mới tới default 70.
            if 'num_inputs' in shlnn_cfgs:
                num_inputs = shlnn_cfgs.get('num_inputs')
            else:
                num_inputs = cfgs.get('shlnn_num_inputs', 70)

        self.num_inputs = int(num_inputs)

        # SỬA (wiring bug): trước đây hidden layer luôn hardcode = 110
        # dù sweep_utils.py có random-search
        # cfgs['SHLNN']['hidden_neurons'] trong [90, 110, 130] -> search
        # với 2/3 giá trị sweep ra không có tác dụng gì. Giờ đọc từ cfgs,
        # fallback về 110 (giá trị paper) nếu không truyền.
        hidden_neurons = int(shlnn_cfgs.get('hidden_neurons', 110))

        # SỬA: thêm Sigmoid ở output layer theo đúng paper Section III-C
        # (xem docstring class, và mục "XUNG ĐỘT VỚI CONFIG" ở trên).
        # Có thể tắt qua cfgs['SHLNN']['sigmoid_output']=False nếu bạn
        # muốn khớp theo config hiện tại (loss_type: CrossEntropy) thay
        # vì paper. Lưu lại làm attribute để trainer_utils.py biết cách
        # chọn loss/target tương ứng (BCELoss+one-hot khi bật, ngược
        # lại CrossEntropyLoss+class-index khi tắt).
        self.use_sigmoid_output = bool(shlnn_cfgs.get('sigmoid_output', True))

        layers = [
            nn.Linear(self.num_inputs, hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_neurons, 2),
        ]
        if self.use_sigmoid_output:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
