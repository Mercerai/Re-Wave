import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)

        return F.relu(x)


class S_Attention_block(nn.Module):
    def __init__(self, in_channels, ker_size=9):
        super(S_Attention_block, self).__init__()
        self.w_layer = nn.Conv1d(in_channels, in_channels, kernel_size=ker_size, stride=1,
                                 groups=in_channels, padding=int((ker_size - 1) // 2))

    def forward(self, x):
        x = self.w_layer(x)

        return torch.sigmoid(x)  # return spatial attention weights


class C_Attention_block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(C_Attention_block, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))

        return x * w  # return the weighted data


class Attention_Block(nn.Module):

    def __init__(self, in_channels, norm_size, ker_size=7, reduction_ratio=16):
        super(Attention_Block, self).__init__()
        self._spatial_block = S_Attention_block(in_channels, ker_size)
        self._channel_block = C_Attention_block(in_channels, reduction_ratio)
        self.norm = nn.LayerNorm(norm_size)

    def forward(self, x):
        x = self.norm(x)
        _spatial_w = self._spatial_block(x)
        x = x * _spatial_w
        out = self._channel_block(x)

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, stride, downsample=None,
                 conv_ker_size=3, ker_size=7, reduction_ratio=16):
        super(BasicBlock, self).__init__()
        padding = int((conv_ker_size - 1) / 2)
        self.conv_block1 = BasicConv1d(in_channels, out_channels, kernel_size=conv_ker_size, padding=padding)
        self.conv_block2 = BasicConv1d(out_channels, out_channels, kernel_size=conv_ker_size,
                                       stride=stride, padding=padding)
        self.attention_block = Attention_Block(out_channels, size, ker_size, reduction_ratio)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)

        out = self.attention_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, size, stride, downsample=None, conv_ker_size=3, ker_size=7,
                 reduction_ratio=16):
        super(Bottleneck, self).__init__()
        self.expansion = 4

        padding = int((conv_ker_size - 1) / 2)
        plains = out_channels // self.expansion
        self.conv_block1 = BasicConv1d(in_channels, plains, kernel_size=1)
        self.conv_block2 = BasicConv1d(plains, plains, stride=stride, kernel_size=conv_ker_size, padding=padding)
        self.conv_block3 = BasicConv1d(plains, out_channels, kernel_size=1)

        self.attention_block = Attention_Block(out_channels, size, ker_size=ker_size, reduction_ratio=reduction_ratio)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)

        out = self.attention_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResSCANet(nn.Module):  # for one channel model

    def __init__(self, in_channels, length=4001):
        super(ResSCANet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=9, stride=3,
                               padding=4)
        length = (length - 1) // 3 + 1
        self.Attention1 = Attention_Block(32, length, 9, 4)
        length = (length - 1) // 2 + 1
        self.layer1 = self._make_layers(BasicBlock, 32, 64, length, 1, 2, 5, 9)  # 64 667
        # self.layer2 = self._make_layers(BasicBlock, 64, 128, 1, 2, 3, 7)  # 128
        length = (length - 1) // 2 + 1
        self.layer3 = self._make_layers(BasicBlock, 64, 160, length, 1, 2, 3, 7)  # 160 334
        length = (length - 1) // 2 + 1
        self.layer4 = self._make_layers(Bottleneck, 160, 304, length, 1, 2, 5, 7)  # 304 167

        self.GLBPooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(304, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

    def _make_layers(self, block_type, in_channels, out_channels, size, num_layers, stride=1, conv_ker_size=3,
                     ker_size=7,
                     reduction_ratio=16):
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                                       nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(
            block_type(in_channels, out_channels, size, stride, downsample=downsample, conv_ker_size=conv_ker_size,
                       ker_size=ker_size, reduction_ratio=reduction_ratio))

        for _ in range(1, num_layers):
            layers.append(
                block_type(out_channels, out_channels, size, stride=1, downsample=None, conv_ker_size=conv_ker_size,
                           ker_size=ker_size, reduction_ratio=reduction_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        # extractor
        #         x = self.norm(x)
        x = self.conv1(x)
        x = self.Attention1(x)
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.GLBPooling(x).view(x.size(0), -1)
        # classifier
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        out = self.fc2(x)

        return out

class ResSCANet6(nn.Module): #for 6 channels model
    def __init__(self, in_channels, length =  4001 ):
        super(ResSCANet6, self).__init__()
        self.atte = Attention_Block(in_channels, length, 9, 1)
        length = (length - 1)//2+1
        self.layer1 = self._make_layers(BasicBlock, 6, 64, length, 1, 2, 9, 9, 8) # 64 2001
        length = (length - 1)//2+1
        self.layer2 = self._make_layers(BasicBlock, 64, 128, length, 1, 2, 5, 7) #128 1001
        length = (length - 1)//2+1
        self.layer3 = self._make_layers(Bottleneck, 128, 256, length, 1, 2, 3, 7) #304 501
        length = (length - 1)//2+1
        self.layer4 = self._make_layers(Bottleneck, 256, 384, length, 1, 2, 3, 7) #384 251

        self.GLBPooling = nn.AdaptiveAvgPool1d(1) #384 1
        self.fc1 = nn.Linear(384, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # print(x.shape)
        x = self.atte(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.GLBPooling(x).view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        out = self.fc2(x)

        return out

    def _make_layers(self, block_type, in_channels, out_channels, size, num_layers, stride=1, conv_ker_size=3, ker_size=7,
                     reduction_ratio=16):
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                                       nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block_type(in_channels, out_channels, size, stride, downsample=downsample, conv_ker_size=conv_ker_size,
                                 ker_size=ker_size, reduction_ratio=reduction_ratio))

        for _ in range(1, num_layers):
            layers.append(block_type(out_channels, out_channels, size, stride=1, downsample=None, conv_ker_size=conv_ker_size,
                                     ker_size=ker_size, reduction_ratio=reduction_ratio))

        return nn.Sequential(*layers)
