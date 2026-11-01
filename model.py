import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class FinalDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = ConvBlock(3, 8)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(8, 16)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(16, 32)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(32, 48)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(80, 32)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(48, 16)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = FinalDecoderBlock(24, 8)

        self.out = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x = self.pool1(x1)

        x2 = self.enc2(x)
        x = self.pool2(x2)

        x3 = self.enc3(x)
        x = self.pool3(x3)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)

        x = self.out(x)
        return x

def combined_loss(pred_logits, target_mask):
    """Combines Cross Entropy with Soft Dice as required for track B."""
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
    
    pred_probs = torch.sigmoid(pred_logits)
    intersection = (pred_probs * target_mask).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3))
    
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    
    return bce + dice.mean()

if __name__ == "__main__":
    model = SmallUNet()

    x = torch.randn(1, 3, 256, 256)
    output = model(x)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Output shape:", output.shape)
    print("Total parameters:", total_params)
    print("Output finite:", torch.isfinite(output).all().item())
    print("Logit range:", output.min().item(), output.max().item())