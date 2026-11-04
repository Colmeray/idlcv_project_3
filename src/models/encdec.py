from torch import nn

from models.unet import conv_block


class EncDec(nn.Module):
    """Simpel encoder-decoder uden skip connections (baseline).

    Samme blokke og kanalantal som UNet, så den eneste forskel er skip connections:
    decoderen skal genskabe detaljer alene ud fra bottleneck-repræsentationen.
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
        self.decoders = nn.ModuleList(conv_block(channels[i], channels[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        for encoder in self.encoders:
            x = self.pool(encoder(x))
        x = self.bottleneck(x)
        for upsample, decoder in zip(self.upsamples, self.decoders):
            x = decoder(upsample(x))
        return self.head(x).squeeze(1)
