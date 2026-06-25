import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) with dense connections as used in ESRGAN.
    """
    def __init__(self, nf=64, gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB) containing 3 dense blocks.
    """
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = DenseBlock(nf, gc)
        self.RDB2 = DenseBlock(nf, gc)
        self.RDB3 = DenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class GeneratorRRDB(nn.Module):
    """
    ESRGAN-backbone Generator with PixelShuffle 2x upsampling.
    Takes 6 channels (stacked visible bands, thermal band, and index bands) 
    and outputs a 3-channel pseudo-RGB image scaled 2x.
    """
    def __init__(self, in_channels=6, out_channels=3, nf=64, gc=32, num_blocks=4):
        super(GeneratorRRDB, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(
            *[RRDB(nf, gc) for _ in range(num_blocks)]
        )
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # Upsampling block using PixelShuffle (2x)
        # We map nf channels to out_channels * (2^2) channels so that PixelShuffle
        # can reshape it to [B, out_channels, 2*H, 2*W]
        self.conv_up = nn.Conv2d(nf, out_channels * (2 ** 2), 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        
        # Restrict output to valid [0, 1] range to represent scaled optical reflectance
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.RRDB_trunk(fea)
        trunk = self.conv_body(trunk)
        out = fea + trunk  # Global residual connection
        out = self.conv_up(out)
        out = self.pixel_shuffle(out)
        return self.sigmoid(out)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator to evaluate visual and color realism at local patch scale.
    Outputs a grid of patch realism scores.
    """
    def __init__(self, in_channels=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final output layer
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
