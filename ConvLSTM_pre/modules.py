import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify(x, patch_size=4):
    batch, seq_len, channels, height, width = x.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    x = x.reshape(
        batch,
        seq_len,
        channels,
        h_patches,
        patch_size,
        w_patches,
        patch_size,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6)
    x = x.reshape(
        batch,
        seq_len,
        h_patches,
        w_patches,
        channels * patch_size * patch_size,
    )
    return x.permute(0, 1, 4, 2, 3).contiguous()


def unpatchify(x, patch_size=4):
    batch, seq_len, patch_dim, h_patches, w_patches = x.shape
    channels = patch_dim // (patch_size * patch_size)
    x = x.reshape(
        batch,
        seq_len,
        channels,
        patch_size,
        patch_size,
        h_patches,
        w_patches,
    )
    x = x.permute(0, 1, 2, 5, 3, 6, 4)
    return x.reshape(
        batch,
        seq_len,
        channels,
        h_patches * patch_size,
        w_patches * patch_size,
    ).contiguous()


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
        )
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_dim_list = hidden_dim_list
        self.cells = nn.ModuleList()

        for layer_idx, hidden_dim in enumerate(hidden_dim_list):
            cur_input_dim = input_dim if layer_idx == 0 else hidden_dim_list[layer_idx - 1]
            self.cells.append(
                ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, padding)
            )

    def forward(self, x, initial_states=None):
        batch, seq_len, _, height, width = x.shape
        device = x.device

        if initial_states is None:
            h_states = [
                torch.zeros(batch, hidden_dim, height, width, device=device)
                for hidden_dim in self.hidden_dim_list
            ]
            c_states = [
                torch.zeros(batch, hidden_dim, height, width, device=device)
                for hidden_dim in self.hidden_dim_list
            ]
        else:
            h_states = [state[0] for state in initial_states]
            c_states = [state[1] for state in initial_states]

        current_input = x
        for layer_idx, cell in enumerate(self.cells):
            h = h_states[layer_idx]
            c = c_states[layer_idx]
            layer_outputs = []
            for t in range(seq_len):
                h, c = cell(current_input[:, t], h, c)
                layer_outputs.append(h.unsqueeze(1))
            current_input = torch.cat(layer_outputs, dim=1)
            h_states[layer_idx] = h
            c_states[layer_idx] = c

        final_states = [(h_states[i], c_states[i]) for i in range(len(self.cells))]
        return current_input, final_states


class PatchEmbedding(nn.Module):
    def __init__(self, patch_dim, hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(patch_dim, hidden_dim, kernel_size=1)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        batch, seq_len, _, h_patches, w_patches = x.shape
        x = x.reshape(batch * seq_len, x.size(2), h_patches, w_patches)
        x = self.conv(x)
        return x.reshape(batch, seq_len, -1, h_patches, w_patches)


class PatchRecovery(nn.Module):
    def __init__(self, hidden_dim, patch_dim):
        super().__init__()
        self.conv = nn.Conv2d(hidden_dim, patch_dim, kernel_size=1)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        batch, seq_len, _, h_patches, w_patches = x.shape
        x = x.reshape(batch * seq_len, x.size(2), h_patches, w_patches)
        x = self.conv(x)
        return x.reshape(batch, seq_len, -1, h_patches, w_patches)


class PrecipConvLSTM(nn.Module):
    def __init__(
        self,
        input_channels=2,
        output_channels=1,
        hidden_dim_list=None,
        kernel_size=5,
        padding=2,
        patch_size=4,
        image_size=(158, 165),
    ):
        super().__init__()
        if hidden_dim_list is None:
            hidden_dim_list = [128, 64, 64]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.patch_size = patch_size
        self.image_size = image_size

        input_patch_dim = input_channels * patch_size * patch_size
        output_patch_dim = output_channels * patch_size * patch_size

        self.patch_embed = PatchEmbedding(input_patch_dim, hidden_dim_list[0])
        self.encoder = ConvLSTM(
            hidden_dim_list[0], hidden_dim_list, kernel_size, padding
        )
        self.forecaster = ConvLSTM(
            hidden_dim_list[-1], hidden_dim_list, kernel_size, padding
        )
        self.patch_recovery = PatchRecovery(hidden_dim_list[-1], output_patch_dim)
        self.output_activation = nn.Softplus()

    def _pad_to_patch_size(self, x):
        height, width = x.shape[-2:]
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, pad=(0, pad_w, 0, pad_h), mode="constant", value=0)

    def forward(self, input_seq, future_steps=10):
        batch = input_seq.size(0)
        device = input_seq.device

        input_seq = self._pad_to_patch_size(input_seq)
        _, _, _, padded_h, padded_w = input_seq.shape

        x = patchify(input_seq, self.patch_size)
        x = self.patch_embed(x)

        _, encoder_final_states = self.encoder(x)
        h_patches, w_patches = x.shape[-2:]

        forecaster_cells = self.forecaster.cells
        h_states = [encoder_final_states[i][0] for i in range(len(forecaster_cells))]
        c_states = [encoder_final_states[i][1] for i in range(len(forecaster_cells))]
        x_t = torch.zeros(
            batch,
            self.forecaster.cells[0].input_dim,
            h_patches,
            w_patches,
            device=device,
        )

        predictions = []
        for _ in range(future_steps):
            layer_input = x_t
            for layer_idx, cell in enumerate(forecaster_cells):
                h_new, c_new = cell(
                    layer_input,
                    h_states[layer_idx],
                    c_states[layer_idx],
                )
                h_states[layer_idx] = h_new
                c_states[layer_idx] = c_new
                layer_input = h_new
            predictions.append(layer_input.unsqueeze(1))
            x_t = layer_input

        predictions = torch.cat(predictions, dim=1)
        predictions = self.patch_recovery(predictions)
        predictions = unpatchify(predictions, self.patch_size)
        predictions = predictions[:, :, :, :padded_h, :padded_w]

        height, width = self.image_size
        predictions = predictions[:, :, :, :height, :width]
        return self.output_activation(predictions)
