import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    """
    编码器：读取 10 帧输入，提取最终隐藏状态。
    输入: (batch_size, 10, 64*64)
    输出: (hidden_state, cell_state), 形状均为 (num_layers, batch_size, hidden_size)
    """

    def __init__(self, input_size=4096, hidden_size=2048, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 初始化 - 与源码一致，使用 uniform 初始化
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -0.01, 0.01)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        x: (B, T, input_size)
        """
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n


class DecoderLSTM(nn.Module):
    """
    解码器：从编码器最终状态解码序列。
    支持条件解码（训练时输入真实帧）和非条件解码（输入零向量）。
    关键修改：只在 t=0 时使用初始状态，t>0 时不传递初始状态。
    """

    def __init__(
        self, input_size=4096, hidden_size=2048, num_layers=1, output_size=4096
    ):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

        # 初始化 - 与源码一致
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.uniform_(param, -0.01, 0.01)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.uniform_(self.fc.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h_init, c_init, target_seq=None, seq_len=10, conditional=False):
        """
        h_init, c_init: (num_layers, B, hidden_size)
        target_seq: (B, seq_len, input_size) - 条件解码时的真实输入
        seq_len: 解码序列长度
        conditional: 是否使用条件解码
        """
        B = h_init.size(1)
        device = h_init.device

        outputs = []
        h_t = h_init
        c_t = c_init

        # 初始输入为零向量
        x_t = torch.zeros(B, 1, self.input_size, device=device)

        for t in range(seq_len):
            # 只在 t=0 时使用初始状态，t>0 时不传递初始状态
            if t == 0:
                out, (h_t, c_t) = self.lstm(x_t, (h_t, c_t))
            else:
                out, (h_t, c_t) = self.lstm(x_t)

            pred = self.sigmoid(self.fc(out.squeeze(1)))
            outputs.append(pred)

            # 条件解码：训练时使用真实上一帧作为输入
            # 非条件解码：使用上一步输出作为下一步输入
            if conditional and target_seq is not None and t > 0:
                x_t = target_seq[:, t - 1 : t, :]
            else:
                x_t = pred.unsqueeze(1)

        return torch.stack(outputs, dim=1)  # (B, seq_len, output_size)


class CompositeModel(nn.Module):
    """
    复合模型：包含一个编码器和两个并行的解码器。
    - Decoder 1: Input Reconstructor (逆序重构)
    - Decoder 2: Future Predictor (未来预测)
    """

    def __init__(self, input_size=4096, hidden_size=2048, num_layers=1):
        super(CompositeModel, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.input_reconstructor = DecoderLSTM(
            input_size, hidden_size, num_layers, input_size
        )
        self.future_predictor = DecoderLSTM(
            input_size, hidden_size, num_layers, input_size
        )

    def forward(self, x_past, x_future, mode="train"):
        """
        x_past: (B, T_past, 1, 64, 64) - 输入序列
        x_future: (B, T_future, 1, 64, 64) - 未来序列
        mode: 'train' 或 'predict'
        """
        B, T_past, C, H, W = x_past.shape
        T_future = x_future.size(1)

        # 展平图像
        x_past_flat = x_past.view(B, T_past, -1)  # (B, 10, 4096)
        x_future_flat = x_future.view(B, T_future, -1)  # (B, 10, 4096)

        # 编码器提取最终状态
        h_n, c_n = self.encoder(x_past_flat)

        # Decoder 1: 逆序重构输入
        x_past_reversed = torch.flip(x_past_flat, dims=[1])  # 逆序
        recon_input = self.input_reconstructor(
            h_n, c_n, target_seq=x_past_reversed, seq_len=T_past, conditional=False
        )

        # Decoder 2: 未来预测
        if mode == "train":
            # 训练时使用 Teacher Forcing（条件解码）
            pred_future = self.future_predictor(
                h_n, c_n, target_seq=x_future_flat, seq_len=T_future, conditional=True
            )
        else:
            # 预测时自回归生成（使用生成的帧作为下一步输入）
            pred_future = self.future_predictor(
                h_n, c_n, target_seq=None, seq_len=T_future, conditional=False
            )

        # 恢复图像形状
        recon_input = recon_input.view(B, T_past, C, H, W)
        pred_future = pred_future.view(B, T_future, C, H, W)

        return recon_input, pred_future
