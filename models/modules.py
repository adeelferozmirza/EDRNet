import torch
from torch import nn
from torch.nn import functional as F

""" Local Context Attention Module"""
class SCA(nn.Module):
    def __init__(self, in_channels):
        super(SCA, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Channel attention
        x_ca = self.channel_attention(x) * x

        # Spatial attention
        x_sca = self.spatial_attention(x_ca) * x_ca

        return x_sca

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class LCA(nn.Module):
    def __init__(self, in_channels):
        super(LCA, self).__init__()
        self.sca = SCA(in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, pred):
        # Upsample pred to match x's spatial dimensions
        # scale_factor_height = x.size(2) / pred.size(2)
        # scale_factor_width = x.size(3) / pred.size(3)
        # self.upsample.scale_factor = (scale_factor_height, scale_factor_width)
        # pred_upsampled = self.upsample(pred)
        x = self.upsample(x)
        # Apply combined spatial and channel attention
        x = self.sca(x)

        # Use pred_upsampled as an additional modulating factor (optional refinement)
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        att = att.expand_as(x)  # Ensure dimension match for broadcasting

        # Element-wise multiplication for attention application
        x_att = x * att

        # Add the original upsampled features (residual connection)
        out = x_att + x  # Previously: out = att_x + residual

        return out


class LCA1(nn.Module):
    def __init__(self):
        super(LCA1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, pred):
        x = self.upsample(x)
        residual = x
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)       
        att_x = x * att

        out = att_x + residual

        return out

class LCA2(nn.Module):
    def __init__(self):
        super(LCA2, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, pred):
        # x = self.upsample(x)
        residual = x
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)      
        att_x = x * att

        out = att_x + residual

        return out

""" Global Context Module"""


class GCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM, self).__init__()
        pool_size = [1, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsampe_scale = [2, 4, 8, 16]
        GClist = []
        GCoutlist = []
        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(4):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)

    def forward(self, x):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
            global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))
        global_context = torch.cat(global_context, dim=1)

        output = []
        for i in range(len(self.GCoutmodel)):
            output.append(self.GCoutmodel[i](global_context))

        return output


""" Adaptive Selection Module"""


class ASM(nn.Module):
    def __init__(self, in_channels, all_channels):
        super(ASM, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.selayer = SELayer(all_channels)

    def forward(self, lc, fuse, gc):
        fuse = self.non_local(fuse)
        fuse = torch.cat([lc, fuse, gc], dim=1)
        # fuse = self.selayer(fuse)
        return fuse


"""
Squeeze and Excitation Layer

https://arxiv.org/abs/1709.01507

"""


class SELayer(nn.Module):
    def __init__(self, channel, reduction=64):
        super(SELayer, self).__init__()
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


"""
Non Local Block

https://arxiv.org/abs/1711.07971
"""


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z