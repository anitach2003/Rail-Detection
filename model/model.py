import torch
from model.backbone import resnet, mobilenet, squeezenet, VisionTransformer
import numpy as np

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(100, 52, 4)):
        # cls_dim: (num_gridding, num_cls_per_lane, num_of_lanes)

        super(parsingNet, self).__init__()
        self.size = size
        self.w = size[1]
        self.h = size[0]
        self.cls_dim = cls_dim 

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
            
        # Classifier head
        self.cls_cat = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, np.prod(cls_dim)),
        )
        initialize_weights(self.cls_cat)

        # -------------------------------
        # ⭐ Row Smoothing 1D Conv (depthwise)
        # -------------------------------
        num_gridding, num_rows, num_lanes = cls_dim
        self.row_smoother = torch.nn.Conv1d(
    in_channels=num_gridding * num_lanes,  # flatten lanes + grids per row
    out_channels=num_gridding * num_lanes,
    kernel_size=3,      # small kernel
    padding=2,          # (kernel_size - 1) // 2 * dilation
    dilation=2,         # look at every 2nd row
    groups=num_gridding * num_lanes,  # depthwise
    bias=False
)
        # -------------------------------


    def forward(self, x):
        # Backbone + pooling
        x4 = self.model(x)
        fea = self.pool(x4).view(-1, 1800)

        # Raw prediction
        group_cat = self.cls_cat(fea).view(-1, *self.cls_dim)
        # shape: [B, G, R, L]

        # -------------------------------
        # ⭐ Apply smoothing across rows
        # -------------------------------
        B, G, R, L = group_cat.shape

        # Rearrange to apply Conv1d along rows (R)
        out = group_cat.permute(0, 2, 1, 3)       # [B, R, G, L]
        out = out.reshape(B, R, G * L).transpose(1, 2)  # [B, G*L, R]

        out = self.row_smoother(out)             # [B, G*L, R]

        # Restore original layout
        out = out.transpose(1, 2).reshape(B, R, G, L)
        out = out.permute(0, 2, 1, 3)            # [B, G, R, L]
        # -------------------------------

        return out
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

