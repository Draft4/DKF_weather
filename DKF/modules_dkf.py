import torch
import random
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=5, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        # 保证卷积后图像的长宽尺寸不变
        padding = kernel_size // 2

        # 用一个卷积层同时计算 i,f,o,g 四个变量
        self.conv = nn.Conv2d(
            self.input_dim + self.hidden_dim,
            4 * self.hidden_dim,
            self.kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        # cur_state 为包含上一时刻 (H_{t-1},c_{t-1}) 的元祖
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.encoder(x)


class InferenceNetwork(nn.Module):
    def __init__(self, e_dim=32, z_dim=32, h_dim=64):
        super(InferenceNetwork, self).__init__()

        self.z_dim = z_dim

        self.rnn = ConvLSTMCell(e_dim + z_dim, h_dim, 5)
        self.mu_net = nn.Conv2d(h_dim, z_dim, 3, padding=1)
        self.logvar_net = nn.Conv2d(h_dim, z_dim, 3, padding=1)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, e_t, z_prev, h_prev, c_prev):
        rnn_input = torch.cat([e_t, z_prev], dim=1)

        h_t, c_t = self.rnn(rnn_input, (h_prev, c_prev))

        mu_qt = self.mu_net(h_t)
        logvar_qt = self.logvar_net(h_t)

        z_t = self.reparameterize(mu_qt, logvar_qt)
        return z_t, mu_qt, logvar_qt, h_t, c_t


class TransitionNetwork(nn.Module):
    def __init__(self, z_dim=32, h_dim=64):
        super(TransitionNetwork, self).__init__()

        self.rnn = ConvLSTMCell(z_dim, h_dim, 5)
        self.mu_net = nn.Conv2d(h_dim, z_dim, 3, padding=1)
        self.logvar_net = nn.Conv2d(h_dim, z_dim, 3, padding=1)

    def forward(self, z_prev, h_prev, c_prev):
        h_t, c_t = self.rnn(z_prev, (h_prev, c_prev))
        mu_pt = self.mu_net(h_t)
        logvar_pt = self.logvar_net(h_t)
        return mu_pt, logvar_pt, h_t, c_t


class ObservationDecoder(nn.Module):
    def __init__(self, z_dim=32):
        super(ObservationDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_t):
        return self.decoder(z_t)


class DeepKalmanFilter(nn.Module):
    def __init__(self):
        super(DeepKalmanFilter, self).__init__()

        self.encoder = ObservationEncoder()
        self.inference_net = InferenceNetwork()
        self.transition_net = TransitionNetwork()
        self.decoder = ObservationDecoder()

        self.z_dim = 32

    def forward(self, x, n_past=10, n_future=10, mode="train", ss_ratio=0.0):
        batch_size = x.size(0)
        device = x.device
        image_size = (16, 16)  # CNN 编码后的特征空间分辨率

        h_q, c_q = self.inference_net.rnn.init_hidden(batch_size, image_size)
        h_p, c_p = self.transition_net.rnn.init_hidden(batch_size, image_size)
        z_t = torch.randn(batch_size, self.z_dim, *image_size, device=device)

        if mode == "train":
            T = n_past + n_future
            kld_loss = 0.0
            recon_preds = []

            for t in range(T):
                # 编码
                e_t = self.encoder(x[:, t, :, :, :])
                # 利用上一步的 z_t 预测
                mu_pt, logvar_pt, h_p, c_p = self.transition_net(z_t, h_p, c_p)
                # 推断
                z_t, mu_qt, logvar_qt, h_q, c_q = self.inference_net(e_t, z_t, h_q, c_q)

                kld_t = 0.5 * torch.mean(
                    torch.exp(logvar_qt - logvar_pt)
                    + (mu_qt - mu_pt).pow(2) / torch.exp(logvar_pt)
                    - 1
                    + logvar_pt
                    - logvar_qt,
                )
                kld_loss += kld_t

                if t > 0 and random.random() < ss_ratio:
                    z_t = mu_pt

                # 解码
                x_hat_t = self.decoder(z_t)
                recon_preds.append(x_hat_t)
            recon_preds = torch.stack(recon_preds, dim=1)
            return recon_preds, kld_loss

        elif mode == "predict":
            predictions = []

            # 阶段 1，历史推断
            for t in range(n_past):
                e_t = self.encoder(x[:, t, :, :, :])
                mu_pt, logvar_pt, h_p, c_p = self.transition_net(z_t, h_p, c_p)
                z_t, mu_qt, logvar_qt, h_q, c_q = self.inference_net(e_t, z_t, h_q, c_q)
            # 阶段 2，未来预测（关闭推断网络）
            for t in range(n_future):
                mu_pt, logvar_pt, h_p, c_p = self.transition_net(z_t, h_p, c_p)
                # 使用先验分布的均值作为最确定的估计，或者也可以使用重采样
                z_t = mu_pt

                x_hat_t = self.decoder(z_t)
                predictions.append(x_hat_t)
            predictions = torch.stack(predictions, dim=1)
            return predictions
