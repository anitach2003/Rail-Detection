import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from model.backbone import resnet, mobilenet, squeezenet, VisionTransformer


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def build_grid_edge_index(height=9, width=25, device='cpu'):
    """Builds a sparse 8-connected grid graph for GATConv."""
    edges = []
    for i in range(height):
        for j in range(width):
            node_id = i * width + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_id = ni * width + nj
                        edges.append((node_id, neighbor_id))
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).T  # [2, num_edges]
    return edge_index


class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(100, 52, 4)):
        # cls_dim: (num_gridding, num_cls_per_lane, num_of_lanes)
        super(parsingNet, self).__init__()
        self.size = size
        self.w = size[1]
        self.h = size[0]
        self.cls_dim = cls_dim
        self.num_nodes = 9 * 25  # 225
        in_features = 8
        hidden_features = 16

        # --- Replace GCNConv with GATConv ---
        self.gc1 = GATConv(in_features, hidden_features, heads=4, concat=True)
        self.gc2 = GATConv(hidden_features * 4, in_features, heads=1, concat=False)
        # -------------------------------------

        # Backbone feature extractor
        if backbone in ['34', '18']:
            self.model = resnet(backbone, pretrained=pretrained)
            self.pool = torch.nn.Conv2d(512, 8, 1)

        elif backbone in ['50', '101']:
            self.model = resnet(backbone, pretrained=pretrained)
            self.pool = torch.nn.Conv2d(2048, 8, 1)

        elif backbone in ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']:
            self.model = mobilenet(backbone, pretrained=pretrained)
            self.pool = torch.nn.Conv2d(1280, 8, 1)

        elif backbone in ['squeezenet1_0', 'squeezenet1_1']:
            self.model = squeezenet(backbone, pretrained=pretrained)
            self.pool = torch.nn.Sequential(
                torch.nn.Conv2d(512, 8, 1),
                torch.nn.AdaptiveAvgPool2d((9, 25)),
            )

        elif backbone in ['vit_b_16']:
            self.model = VisionTransformer(backbone, pretrained=pretrained)
            self.pool = torch.nn.Sequential(
                torch.nn.Linear(768, 1800),
            )

        # Classification head
        self.cls_cat = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, np.prod(cls_dim)),
        )

        initialize_weights(self.cls_cat)

    def forward(self, x):
        # Extract backbone features
        x4 = self.model(x)
        fea = self.pool(x4)

        # Downsample to 9x25 grid
        fea = F.adaptive_avg_pool2d(fea, (9, 25))
        fea = fea.view(fea.size(0), 8, -1).permute(0, 2, 1)  # [B, 225, 8]

        # Build sparse adjacency for 9x25 grid
        device = fea.device
        edge_index = build_grid_edge_index(9, 25, device)

        outputs = []
        for b in range(fea.size(0)):
            node_features = fea[b]  # [225, 8]
            x1 = F.relu(self.gc1(node_features, edge_index))
            x2 = self.gc2(x1, edge_index)
            outputs.append(x2)

        fea = torch.stack(outputs, dim=0)  # [B, 225, 8]
        fea = fea.reshape(fea.size(0), -1)
        group_cat = self.cls_cat(fea).view(-1, *self.cls_dim)

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
        elif isinstance(m, torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unknown module', m)
