import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=4):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_ch, 32)    # [B,32,128,128]
        self.pool1 = nn.MaxPool2d(2)          # [B,32,64,64]
        self.down2 = DoubleConv(32, 64)       # [B,64,64,64]
        self.pool2 = nn.MaxPool2d(2)          # [B,64,32,32]
        self.down3 = DoubleConv(64, 128)      # [B,128,32,32]
        self.pool3 = nn.MaxPool2d(2)          # [B,128,16,16]
        
        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)# [B,256,16,16]

        self.cls_conv = nn.Sequential(
            DoubleConv(64, 128),  # [B,128,32,32]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Decoder
        self.up1 = self._upsample(256, 128)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = self._upsample(128, 64)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = self._upsample(64, 32)
        self.conv3 = DoubleConv(64, 32)  
        
        # Final
        self.final = nn.Conv2d(32, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def _upsample(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)        # [B,32,128,128]
        x2 = self.pool1(x1)       # [B,32,64,64]
        x2 = self.down2(x2)       # [B,64,64,64]
        x3 = self.pool2(x2)       # [B,64,32,32]
        x3 = self.down3(x3)       # [B,128,32,32]
        x4 = self.pool3(x3)       # [B,128,16,16]
        
        # Bottleneck
        x4 = self.bottleneck(x4)  # [B,256,16,16]
        
        # Decoder
        x = self.up1(x4)          # [B,128,32,32]
        x = torch.cat([x3, x], dim=1)  # [B,256,32,32]
        x = self.conv1(x)         # [B,128,32,32]
        
        x = self.up2(x)           # [B,64,64,64]
        x = torch.cat([x2, x], dim=1)  # [B,128,64,64]
        x = self.conv2(x)         # [B,64,64,64]
        
        x = self.up3(x)           # [B,32,128,128]
        x = torch.cat([x1, x], dim=1)  # [B,64,128,128]
        x = self.conv3(x)         # [B,32,128,128]
        
        heatmap = self.sigmoid(self.final(x))

        # cls_output = self.classifier(x4)
        # cls_output = self.classifier(heatmap)
        cls_output = self.cls_conv(x2)

        return heatmap, cls_output  # [B,4,128,128]

if __name__ == "__main__":
    model = UNet()
    x = torch.randn(2, 3, 128, 128)
    heatmap, cls_output = model(x)
    print("Heatmap shape:", heatmap.shape)  # [1, num_keypoints, 128, 128]
    print("Classification output shape:", cls_output.shape)  # [1, 1]
    print("Classification output:", cls_output)