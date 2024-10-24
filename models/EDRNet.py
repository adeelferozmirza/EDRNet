import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights

from models.modules import LCA, GCM, ASM, LCA2, LCA1
# from modules import LCA, GCM, ASM, LCA2, LCA1




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class DeepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DeepConv2D, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer with different kernel size and dilation
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=2, dilation=2)  
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Third convolutional layer with further variation
        self.conv3 = nn.Conv2d(out_channels, out_channels, 5, 1, padding=2)  
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x
    
class DNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DNET, self).__init__()
        self.deep = DeepConv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3,stride=1, padding=1)
        self.resnet1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1,padding=1)
        self.resnet2 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.resnet3 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.separated = DeepConv2D(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.deep(x)
        # print('x1:', x1.shape)
        x2 = self.conv(x)
        # print('x2:', x2.shape)
        x3 = self.resnet1(x)
        x4 = self.resnet2(x3)
        x5 = self.resnet3(x4)
        # print('x5:', x5.shape)
        x6 = self.separated(x)
        # print('x6:', x6.shape)
        out = x1 + x2 + x3 + x4 + x5 + x6
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        if in_channels // 8 < 1:
            self.inter_channels = 1
        else:
            self.inter_channels = in_channels // 8
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.inter_channels, in_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        avg_out = self.fc2(self.relu(avg_out))
        max_out = self.fc1(self.max_pool(x))
        max_out = self.fc2(self.relu(max_out))
        out = avg_out + max_out
        return self.sigmoid(out)


class CAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CAConv, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.conv2 = ConvBlock(in_channels // 4, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.dropout = nn.Dropout2d(0.1)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1)
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        ca = self.channel_attention(x)
        x = x * ca

        return x

class EDRNet(nn.Module):
    def __init__(self, num_classes):
        super(EDRNet, self).__init__()

        # Load pretrained EfficientNet_V2_L
        efficient_net = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        # efficient_net = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)        
        # Encoder - utilizing the features of EfficientNet
        self.encoder1_conv = efficient_net.features[0]  # First conv layer
        self.encoder1_relu = nn.ReLU(inplace=True)      # ReLU activation (inplace saves memory)
        
        # EfficientNet V2 layers are structured differently, need to carefully select them
        self.encoder2 = efficient_net.features[1:3]     # Corresponds to stages 2 and 3 in EfficientNet
        self.encoder3 = efficient_net.features[3:6]     # Stages 4, 5, 6
        self.encoder4 = efficient_net.features[6:10]    # Stages 7 to 10
        self.encoder5 = efficient_net.features[10:]     # Remaining stages
        
        # Decoder
        self.decoder5 = DecoderBlock(in_channels=1280, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=2048, out_channels=512)
        self.decoder3 = DecoderBlock(in_channels=800, out_channels=256)
        self.decoder2 = DecoderBlock(in_channels=368, out_channels=128)
        self.decoder1 = DecoderBlock(in_channels=216, out_channels=64)
        self.dnet = DNET(64, 64)
        self.outconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        self.caconv2 = CAConv(128, 1)
        self.caconv3 = CAConv(256, 1)
        self.caconv4 = CAConv(512, 1)
        self.caconv5 = CAConv(512, 1)

        # local context attention module
        self.lca1 = LCA2()
        self.lca2 = LCA2()
        self.lca3 = LCA1()
        self.lca4 = LCA(1280)

        # global context module
        self.gcm = GCM(1280, 64)

        # adaptive selection module
        self.asm4 = ASM(512, 2048)
        self.asm3 = ASM(512, 256)
        self.asm2 = ASM(256, 256)
        self.asm1 = ASM(128, 192)

    def forward(self, x):
        # x 224
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_relu(e1)
        # print('e1:', e1.shape)
        e2 = self.encoder2(e1)
        # print('e2:', e2.shape)
        e3 = self.encoder3(e2)  # 28
        # print('e3:', e3.shape)
        e4 = self.encoder4(e3)  # 14
        # print('e4:', e4.shape)
        e5 = self.encoder5(e4)  # 7
        
        # print('e5:', e5.shape)
        global_contexts = self.gcm(e5)
        
        d5 = self.decoder5(e5)  # 14
        # print('d5:', d5.shape)
        out5 = self.caconv5(d5)
        # print('out5:', out5.shape)
        lc4  = self.lca4(e4, out5)
        # print('lc4:', lc4.shape)
        gc4 = global_contexts[0]
        comb4 = self.asm4(lc4, d5, gc4)
        # print('comb4:', comb4.shape)

        d4 = self.decoder4(comb4)  # 28
        # print('d4:', d4.shape)
        out4 = self.caconv4(d4)
        # print('out4:', out4.shape)
        lc3 = self.lca3(e3, out4)
        # print('lc3:', lc3.shape)
        gc3 = global_contexts[1]
        comb3 = self.asm3(lc3, d4, gc3)
        # print('comb3:', comb3.shape)


        d3 = self.decoder3(comb3)  # 56
        # print('d3:', d3.shape)
        out3 = self.caconv3(d3)
        # print('out3:', out3.shape)
        lc2 = self.lca2(e2, out3)
        # print('lc2:', lc2.shape)
        gc2 = global_contexts[2]
        comb2 = self.asm2(lc2, d3, gc2)
        # print('comb2:', comb2.shape)
        
        d2 = self.decoder2(comb2)  # 128
        out2 = self.caconv2(d2)
        lc1 = self.lca1(e1, out2)
        gc1 = global_contexts[3]
        comb1 = self.asm1(lc1, d2, gc1)

        d1 = self.decoder1(comb1)  # 224*224*64
        d1 = self.dnet(d1)
        out1 = self.outconv(d1)  # 224

        return torch.sigmoid(out1)


