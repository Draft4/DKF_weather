import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self.Wxi = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.Whi = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        nn.init.xavier_normal_(self.Wxi.weight)
        nn.init.xavier_normal_(self.Whi.weight)
        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Whi.bias)

        self.Wxf = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.Whf = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        nn.init.xavier_normal_(self.Wxf.weight)
        nn.init.xavier_normal_(self.Whf.weight)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Whf.bias)

        self.Wxo = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.Who = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        nn.init.xavier_normal_(self.Wxo.weight)
        nn.init.xavier_normal_(self.Who.weight)
        nn.init.zeros_(self.Wxo.bias)
        nn.init.zeros_(self.Who.bias)

        self.Wxc = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.Whc = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        nn.init.xavier_normal_(self.Wxc.weight)
        nn.init.xavier_normal_(self.Whc.weight)
        nn.init.zeros_(self.Whc.bias)
        nn.init.zeros_(self.Whc.bias)

        self.Wci = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.Wcf = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.Wco = nn.Conv2d(hidden_dim, hidden_dim, 1)
        nn.init.xavier_normal_(self.Wci.weight)
        nn.init.xavier_normal_(self.Wcf.weight)
        nn.init.xavier_normal_(self.Wco.weight)
        nn.init.zeros_(self.Wci.bias)
        nn.init.zeros_(self.Wcf.bias)
        nn.init.zeros_(self.Wco.bias)

    def forward(self, x, h_prev, c_prev):
        batch_size, _, H, W = x.shape

        i = self.Wxi(x) + self.Whi(h_prev) + self.Wci(c_prev)
        f = self.Wxf(x) + self.Whf(h_prev) + self.Wcf(c_prev)
        o = self.Wxo(x) + self.Who(h_prev) + self.Wco(c_prev)
        g = torch.tanh(self.Wxc(x) + self.Whc(h_prev))

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, kernel_size=3, padding=1):
        super(ConvLSTM, self).__init__()
        self.num_layers = len(hidden_dim_list)
        self.hidden_dim_list = hidden_dim_list

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            cur_in = input_dim if i == 0 else hidden_dim_list[i - 1]
            self.cells.append(
                ConvLSTMCell(cur_in, hidden_dim_list[i], kernel_size, padding)
            )

    def forward(self, x, initial_states=None):
        batch, seq_len, _, H, W = x.shape
        device = x.device

        if initial_states is None:
            h_states = [
                torch.zeros(batch, ch, H, W, device=device)
                for ch in self.hidden_dim_list
            ]
            c_states = [
                torch.zeros(batch, ch, H, W, device=device)
                for ch in self.hidden_dim_list
            ]
        else:
            h_states, c_states = zip(*initial_states)
            h_states = list(h_states)
            c_states = list(c_states)

        current_input = x
        for layer_idx, cell in enumerate(self.cells):
            layer_outputs = []
            h, c = h_states[layer_idx], c_states[layer_idx]
            for t in range(seq_len):
                h, c = cell(current_input[:, t], h, c)
                layer_outputs.append(h.unsqueeze(1))
            # 该层的输出作为下一层的输出 (batch, seq_len, hidden, H, W)
            current_input = torch.cat(layer_outputs, dim=1)
            # 更新每一层的最终状态
            h_states[layer_idx], c_states[layer_idx] = h, c
        last_layer_output = current_input
        final_states = [(h_states[i], c_states[i]) for i in range(self.num_layers)]
        return last_layer_output, final_states


class PatchEmbedding(nn.Module):
    """
    4x4 Patch Embedding: 将 patch 序列通过线性层映射到隐藏维度
    输入: (batch, seq_len, patch_dim, h_patches, w_patches)
    输出: (batch, seq_len, hidden_dim, h_patches, w_patches)
    """

    def __init__(self, patch_dim, hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(patch_dim, hidden_dim, kernel_size=1)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x: (batch, seq_len, patch_dim, h_patches, w_patches)
        batch, seq_len, patch_dim, h_patches, w_patches = x.shape
        # 合并 batch 和 seq_len 维度
        x = x.reshape(batch * seq_len, patch_dim, h_patches, w_patches)
        x = self.conv(x)
        # 恢复维度
        x = x.reshape(batch, seq_len, -1, h_patches, w_patches)
        return x


class PatchRecovery(nn.Module):
    """
    4x4 Patch Recovery: 将隐藏状态映射回 patch 维度
    输入: (batch, seq_len, hidden_dim, h_patches, w_patches)
    输出: (batch, seq_len, patch_dim, h_patches, w_patches)
    """

    def __init__(self, hidden_dim, patch_dim):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, patch_dim, kernel_size=1)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim, h_patches, w_patches)
        batch, seq_len, hidden_dim, h_patches, w_patches = x.shape
        x = x.reshape(batch * seq_len, hidden_dim, h_patches, w_patches)
        x = self.conv(x)
        x = x.reshape(batch, seq_len, -1, h_patches, w_patches)
        return x


class EncodingForecastingConvLSTM(nn.Module):
    def __init__(self, patch_dim, hidden_dim_list, kernel_size=3, padding=1):
        super().__init__()
        # Patch Embedding: 将 patch_dim 映射到第一层的 hidden_dim
        self.patch_embed = PatchEmbedding(patch_dim, hidden_dim_list[0])
        self.encoder = ConvLSTM(
            hidden_dim_list[0], hidden_dim_list, kernel_size, padding
        )
        # Forecaster 输入维度应为最后一层的隐藏维度
        self.forecaster = ConvLSTM(
            hidden_dim_list[-1], hidden_dim_list, kernel_size, padding
        )
        # Patch Recovery: 将最后一层的 hidden_dim 映射回 patch_dim
        self.patch_recovery = PatchRecovery(hidden_dim_list[-1], patch_dim)

    def forward(self, input_seq, future_steps):
        # input_seq: (batch, seq_len, patch_dim, h_patches, w_patches)
        batch = input_seq.shape[0]
        H = input_seq.shape[3]
        W = input_seq.shape[4]
        device = input_seq.device

        # Patch Embedding
        x = self.patch_embed(input_seq)
        # x: (batch, seq_len, hidden_dim, h_patches, w_patches)

        # Encoder
        _, encoder_final_states = self.encoder(x)
        forecaster_states = encoder_final_states

        # Forecaster 第一层的输入维度是最后一层的隐藏维度
        in_dim = self.forecaster.cells[0].input_dim
        x = torch.zeros(batch, in_dim, H, W, device=device)

        predictions = []

        forecaster_cells = self.forecaster.cells
        num_layers = len(forecaster_cells)
        h_states = [forecaster_states[i][0] for i in range(num_layers)]
        c_states = [forecaster_states[i][1] for i in range(num_layers)]

        for step in range(future_steps):
            for layer_idx, cell in enumerate(forecaster_cells):
                if layer_idx == 0:
                    h_new, c_new = cell(x, h_states[layer_idx], c_states[layer_idx])
                else:
                    h_new, c_new = cell(h_new, h_states[layer_idx], c_states[layer_idx])
                h_states[layer_idx] = h_new
                c_states[layer_idx] = c_new

            predictions.append(h_new.unsqueeze(1))
            x = h_new

        # (batch, future_steps, hidden_dim, H, W)
        predictions = torch.cat(predictions, dim=1)
        # Patch Recovery: 映射回 patch_dim
        predictions = self.patch_recovery(predictions)
        return predictions
