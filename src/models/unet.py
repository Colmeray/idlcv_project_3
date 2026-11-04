import torch
from torch import nn


def conv_block(in_channels, out_channels):
    """To 3x3-convs med padding=1, så den rumlige størrelse bevares."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net (Ronneberger et al. 2015): maxpool ned, transposed conv op, skip connections via concat.

    Kanaler på niveau i er base_channels * 2**i. Input-størrelsen skal være delelig med 2**depth.
    Output er logits (B x H x W); sigmoid og threshold anvendes først ved evaluering.
    """

    def __init__(self, in_channels=3, base_channels=32, depth=4):
        super().__init__()
        channels = [base_channels * 2**i for i in range(depth + 1)]
        self.encoders = nn.ModuleList(
            conv_block(c_in, c_out) for c_in, c_out in zip([in_channels] + channels[:-2], channels[:-1])
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(channels[-2], channels[-1])
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2) for i in reversed(range(depth))
        )
        self.decoders = nn.ModuleList(conv_block(2 * channels[i], channels[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = decoder(torch.cat([upsample(x), skip], dim=1))
        return self.head(x).squeeze(1)
