r""" The proposed CRNet
"""

import torch
import torch.nn as nn
from collections import OrderedDict

#from utils import logger

__all__ = ["crnet"]

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=True)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class BigConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        
        super(BigConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=2, dilation=2, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class RefineBlock(nn.Module):
    def __init__(self):
        super(RefineBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 8, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x3_2', ConvBN(8, 16, 3)),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv3x3_3', ConvBN(16, 2, 3)),
        ]))
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        
        identity = self.identity(x)
        
        out = self.path1(x)
        out = self.relu(out + identity)
        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
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

#CSINet

class CSIEncoder(nn.Module):
    
    def __init__(self, reduction=4):
        super(CSIEncoder, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.dropout = nn.Dropout(0.15)
        self.encoder_fc = nn.Linear(total_size, 2048 // reduction)
        self.jigsaw_fc = nn.Linear(total_size, 16)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        n=x.size(0)
        
        encode1 = self.encoder1(x) # same size as x
        
        out1 = self.encoder_fc(encode1.view(n,-1)) #[batchsize,2048 // reduction]
        out2 = self.jigsaw_fc(encode1.view(n,-1)) #[batchsize,81]
        
        out2 = out2.reshape([n,4,4])
        out2 = self.softmax(out2)
        
        return out1,out2

class CSIDecoder(nn.Module):
    
    def __init__(self, reduction=4):
        super(CSIDecoder, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.decoder_fc = nn.Linear(2048 // reduction, total_size)
        decoder = OrderedDict([
            ("RFBlock1", RefineBlock()),
            ("RFBlock2", RefineBlock())
        ])
        self.decoder2 = ConvBN(in_channel, 2, 3)
        self.sigmoid = nn.Sigmoid()
        self.decoder_feature = nn.Sequential(decoder)

    def forward(self, x):
        out = self.decoder_fc(x).view(-1, 2, 32, 32)
        out = self.decoder_feature(out)
        out = self.decoder2(out)
        out = self.sigmoid(out)
        return out

class CSINet(nn.Module):
    def __init__(self, reduction=4):
        super(CSINet, self).__init__()
        self.csiencoder=CSIEncoder(reduction)
        self.csidecoder=CSIDecoder(reduction)

    def forward(self, x):
        
        n, c, h, w = x.detach().size() 
        out,jigsaw_out= self.csiencoder(x)
        out= self.csidecoder(out)
        return out


## CRNet
# 
#  
class CREncoder(nn.Module):
    
    def __init__(self, reduction=4):
        super(CREncoder, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 2, 3)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        self.dropout = nn.Dropout(0.15)
        self.encoder_fc = nn.Linear(total_size, 2048 // reduction)
        self.jigsaw_fc = nn.Linear(total_size, 16)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        n=x.size(0)
        encode1 = self.encoder1(x) 
        encode2 = self.encoder2(x) 
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)

        out1 = self.encoder_fc(out.view(n,-1)) 
        out2 = self.jigsaw_fc(out.view(n,-1)) 
        out2 = out2.reshape([n,4,4])
        out2 = self.softmax(out2)
        
        return out1,out2

class CRDecoder(nn.Module):
    
    def __init__(self, reduction=4):
        super(CRDecoder, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.decoder_fc = nn.Linear(2048 // reduction, total_size)

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder2 = ConvBN(in_channel, 2, 3)
        self.sigmoid = nn.Sigmoid()
        self.decoder_feature = nn.Sequential(decoder)



    def forward(self, x):
        
        out = self.decoder_fc(x).view(-1, 2, 32, 32)
        out = self.decoder_feature(out)
        out = self.sigmoid(out)

        return out

class CRNet(nn.Module):
    def __init__(self, reduction=4):
        super(CRNet, self).__init__()
        self.crencoder=CREncoder(reduction)
        self.crdecoder=CRDecoder(reduction)

    def forward(self, x):
        
        n, c, h, w = x.detach().size()
        out,jigsaw_out= self.crencoder(x)
        out= self.crdecoder(out)
        return out

## CLNet

class CLEncoder(nn.Module):
    
    def __init__(self, reduction=4):
        super(CLEncoder, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.encoder2 = ConvBN(in_channel, 32, 1)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x1_bn", ConvBN(34, 2, 1)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))
        
        self.sa = SpatialGate()
        self.se = SELayer(32)
        self.encoder_fc = nn.Linear(total_size, 2048 // reduction)
        self.jigsaw_fc = nn.Linear(total_size, 16)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        n=x.size(0)
        encode1 = self.encoder1(x) 
        encode1 = self.sa(encode1)
        encode2 = self.encoder2(x) 
        encode2 = self.se(encode2)

        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out1 = self.encoder_fc(out.view(n,-1)) 
        out2 = self.jigsaw_fc(out.view(n,-1)) 
        out2 = out2.reshape([n,4,4])
        out2 = self.softmax(out2)
        
        return out1,out2

class CLDecoder(nn.Module):
    
    def __init__(self, reduction=4):
        super(CLDecoder, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        self.decoder_fc = nn.Linear(2048 // reduction, total_size)

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder2 = ConvBN(in_channel, 2, 3)
        self.sigmoid = nn.Sigmoid()
        self.decoder_feature = nn.Sequential(decoder)



    def forward(self, x):
        
        out = self.decoder_fc(x).view(-1, 2, 32, 32)
        out = self.decoder_feature(out)
        out = self.sigmoid(out)

        return out

class CLNet(nn.Module):
    def __init__(self, reduction=4):
        super(CLNet, self).__init__()
        self.clencoder=CLEncoder(reduction)
        self.cldecoder=CLDecoder(reduction)

    def forward(self, x):
        
        n, c, h, w = x.detach().size() 
        out,jigsaw_out= self.clencoder(x)
        out= self.cldecoder(out)
        return out


def crnet(reduction=4):
    r""" 
    config zoom for u to choose different network.
    """
    #model = CRNet(reduction=reduction) 
    model = CLNet(reduction=reduction)
    #model = CSINet(reduction=reduction)
    return model





