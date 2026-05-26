import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


class SEBlock(nn.Module):
    """SE注意力模块"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """残差块 - 增强梯度流动"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels, reduction=16)
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.block(x)


class DoubleBranchCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DoubleBranchCNN, self).__init__()

        # 双分支特征提取（使用残差块）
        self.branch1 = nn.Sequential(
            ResidualBlock(3, 64, stride=2, dropout=0.1),
            ResidualBlock(64, 128, stride=2, dropout=0.15),
            ResidualBlock(128, 256, stride=2, dropout=0.2),
            ResidualBlock(256, 512, stride=2, dropout=0.25)
        )

        self.branch2 = nn.Sequential(
            ResidualBlock(3, 64, stride=2, dropout=0.1),
            ResidualBlock(64, 128, stride=2, dropout=0.15),
            ResidualBlock(128, 256, stride=2, dropout=0.2),
            ResidualBlock(256, 512, stride=2, dropout=0.25)
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CBAM(512)
        )
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头（增强版）
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
        # 改进的权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming初始化 + Xavier初始化组合"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 确保输入有6个通道
        if x.size(1) != 6:
            raise ValueError(f"Expected 6 input channels, got {x.size(1)}")
            
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        # 双分支特征提取
        feat1 = self.branch1(x1)
        feat2 = self.branch2(x2)

        # 特征融合
        feat = torch.cat([feat1, feat2], dim=1)
        feat = self.fusion_conv(feat)

        # 全局池化和分类
        feat = self.avg_pool(feat).view(feat.size(0), -1)
        feat = self.fc1(feat)
        feat = self.fc2(feat)
        out = self.classifier(feat)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    model = DoubleBranchCNN(num_classes=2).to(device)
    print(f"✅ 模型创建成功！")

    print("\n📌 模型结构：")
    print(model)

    batch_size = 4
    test_input = torch.randn(batch_size, 6, 224, 224).to(device)
    with torch.no_grad():
        output = model(test_input)

    print(f"\n📊 前向传播测试：")
    print(f"   输入形状：{test_input.shape}")
    print(f"   输出形状：{output.shape} (应为 {batch_size}×2)")
    print(f"   输出示例：{output.argmax(dim=1).cpu().numpy()}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📈 模型参数量：")
    print(f"   总参数量：{total_params / 1e6:.2f}M")
    print(f"   可训练参数：{trainable_params / 1e6:.2f}M")

    print("\n✅ 图像模型测试全部通过！")