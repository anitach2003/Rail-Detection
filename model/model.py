import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from model.backbone import resnet, mobilenet, squeezenet, VisionTransformer
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return feat_align + feat_arm
        
class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x,x2,x3,x4
class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=0):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation = 1, bias = False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.upsample = upsample

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [conv_bn_relu(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(conv_bn_relu(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [conv_bn_relu(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(conv_bn_relu(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class SegmentationHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=4):
        super(SegmentationHead, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.upsampling(x)
        return x

class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(100, 52, 4), use_aux=False):
        # cls_dim: (num_gridding, num_cls_per_lane, num_of_lanes)

        super(parsingNet, self).__init__()
        self.size = size
        self.w = size[1]
        self.h = size[0]
        self.cls_dim = cls_dim 
        self.num_nodes = 9 * 25  # 225
        in_features = 8
        hidden_features = 16
        self.gc1 = GCNConv(in_features, hidden_features)
        self.gc2 = GCNConv(hidden_features, in_features)
        self.use_aux=use_aux
        # input : nchw,
        # 1/32,
        # 288,800 -> 9,25
        if backbone in ['34','18']:
            self.model = resnet(backbone, pretrained=pretrained)
            self.pool = torch.nn.Conv2d(512,8,1)

        if backbone in ['50','101']:
            self.model = resnet(backbone, pretrained=pretrained)
            self.pool = torch.nn.Conv2d(2048,8,1)

        if backbone in ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']:
            self.model = mobilenet(backbone, pretrained=pretrained)
            self.pool = torch.nn.Conv2d(1280,8,1)

        if backbone in ['squeezenet1_0', 'squeezenet1_1',]:
            self.model = squeezenet(backbone, pretrained=pretrained)
            self.pool = torch.nn.Sequential(
                            torch.nn.Conv2d(512,8,1),
                            torch.nn.AdaptiveAvgPool2d((9, 25)),
                            )
            
        if backbone in ['vit_b_16', ]:
            self.model = VisionTransformer(backbone, pretrained=pretrained)
            self.pool = torch.nn.Sequential(
                            torch.nn.Linear(768, 1800),
                            )
            
        # input: 9,25,8 = 1800
        # output: (gridding_num+1) * sample_rows * 4
        # 56+1 * 42 * 4
        self.cls_cat = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, np.prod(cls_dim)),
        )

        initialize_weights(self.cls_cat)
        self.total_dim = np.prod(cls_dim)
        self.p5 = conv_bn_relu(512, 128) if backbone in ['34','18'] else conv_bn_relu(2048, 128)
        self.p4 = FPNBlock(128, 256) if backbone in ['34','18'] else FPNBlock(128, 1024)
        self.p3 = FPNBlock(128, 128) if backbone in ['34','18'] else FPNBlock(128, 512)
        self.p2 = FPNBlock(128, 64) if backbone in ['34','18'] else FPNBlock(128, 256)
        self.smooth5 = SegmentationBlock(128, 128, n_upsamples=3)
        self.smooth4 = SegmentationBlock(128, 128, n_upsamples=2)
        self.smooth3 = SegmentationBlock(128, 128, n_upsamples=1)
        self.smooth2 = SegmentationBlock(128, 128, n_upsamples=0)

        # Final layers output : n, num_of_lanes+1, h, w
        self.finallayer = SegmentationHead(128*4, cls_dim[-1]+1)

        initialize_weights(self.p5, self.p4, self.p3, self.p2,
                            self.smooth5, self.smooth4, self.smooth3, self.smooth2, self.finallayer)
    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        c2, c3, c4, c5 = self.model(x)
        if self.use_aux:
            
            p5 = self.p5(c5)
            p4 = self.p4(p5, c4)
            p3 = self.p3(p4, c3)
            p2 = self.p2(p3, c2)
            p5 = self.smooth5(p5)
            p4 = self.smooth4(p4)
            p3 = self.smooth3(p3)
            p2 = self.smooth2(p2)
            seg = self.finallayer(torch.cat([p5, p4, p3, p2], dim=1))
        else:
            seg=None

        x4 = self.model(c5)

        fea = self.pool(x4).view(-1, 1800)

        group_cat = self.cls_cat(fea).view(-1, *self.cls_dim) 
        if self.use_aux:
            return group_cat, seg
        return group_cat

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)


#a=parsingNet(size=(288, 800), pretrained=True, backbone='18', cls_dim=(100, 52, 4), use_aux=True)
#b=torch.rand(1,3,288,800)
#c=a(b)









