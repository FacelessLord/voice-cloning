import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiperiodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, return_features=False):
        y_ds = []
        fmaps = []

        for i, d in enumerate(self.discriminators):
            if return_features:
                y_d, fmap = d(y, return_features=True)
            else:
                y_d = d(y)
                fmap = []

            y_ds.append(y_d)
            fmaps.append(fmap)

        return y_ds, fmaps


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        norm_f = nn.utils.parametrizations.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
        ])

        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, input, return_features=False):
        x = input
        fmap = []

        # Reshape 1D audio [B, 1, T] to 2D [B, 1, T//period, period]
        b, c, t = input.shape

        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        if return_features:
            return x, fmap

        return x
