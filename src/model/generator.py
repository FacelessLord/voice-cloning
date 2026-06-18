import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class SimplifiedVITSGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_channels=192):
        super().__init__()
        self.hidden_channels = hidden_channels
        assert hidden_channels != vocab_size

        # 1. Text encoder
        self.emb = nn.Embedding(vocab_size, hidden_channels)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
        )

        # 2. Duration Predictor (Predicts how long each phoneme lasts)
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, 3, padding=1),
        )

        # 3. HiFi-GAN Decoder (Upsamples to 16kHz audio)
        self.decoder = HiFiGANGenerator(
            initial_channel=hidden_channels,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[1, 3, 5],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
        )

    def forward(self, input_ids, real_durations=None, training=True):
        # 1. Encode Text
        # input_ids shape: [batch, seq_len]
        x = self.emb(input_ids)  # [batch, seq_len, hidden]
        x = x.transpose(1, 2)  # [batch, hidden, seq_len]
        x = self.encoder(x)  # [batch, hidden, seq_len]

        log_dur_predict = self.duration_predictor(x).squeeze(1)  # [batch, seq_len]

        b, c, t = x.shape

        if training:
            durs = real_durations  # get_aligned_durations(t, mel_spec_target, b, x.device)
        else:
            durs = torch.round(torch.exp(log_dur_predict)).clamp(min=1).long()

        # max_total_frames = 500
        # if durs.sum() > max_total_frames:
        #     durs = (durs.float() / durs.sum() * max_total_frames).round().clamp(min=1).long()

        expanded_list = []
        for i in range(b):
            exp_b = torch.repeat_interleave(x[i], durs[i], dim=-1)
            expanded_list.append(exp_b.transpose(0, 1))

        x_upsampled = rnn_utils.pad_sequence(expanded_list, batch_first=True, padding_value=0.0).transpose(1, 2)

        # 4. Decode to Audio
        audio_generated = self.decoder(x_upsampled)

        # kl_loss = -0.5 * torch.sum(1 + logs_p - m_p.pow(2) - logs_p.exp()) / (batch_size * text_len)

        log_durs = torch.log(durs.float() + 1e-8)
        dur_loss = F.mse_loss(log_dur_predict, log_durs)

        return {
            "audio_generated": audio_generated,
            "dur_loss": dur_loss,
            "durs": durs
        }


class HiFiGANGenerator(nn.Module):
    def __init__(
            self,
            initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_kernel_sizes)

        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2
                )
            )
        self.resblocks = nn.ModuleList()

        ch = upsample_initial_channel // 2
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            resblocks_list = nn.ModuleList()
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                resblocks_list.append(nn.Conv1d(ch, ch, k, 1, dilation=d, padding=(k * d - d) // 2))
            self.resblocks.append(resblocks_list)

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, input):
        value = self.conv_pre(input)
        for i in range(self.num_upsamples):
            value = F.leaky_relu(value, 0.1)
            value = self.ups[i](value)
            xs = 0
            for j in range(self.num_kernels):
                xs += F.leaky_relu(self.resblocks[i][j](value), 0.1)
            value = xs / self.num_kernels
        value = F.leaky_relu(value)
        value = torch.tanh(self.conv_post(value))

        return value
