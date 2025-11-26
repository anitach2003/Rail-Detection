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

class parsingNet(nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='18',
                 cls_dim=(200, 52, 4), hidden_dim=256, num_heads=8,
                 num_encoder_layers=4, num_decoder_layers=4):
        super(parsingNet, self).__init__()

        self.size = size
        self.h, self.w = size
        self.cls_dim = cls_dim
        self.num_rows, self.num_cols, self.num_lanes = cls_dim

        # -----------------------------
        # Backbone + pooling (your code)
        # -----------------------------
        if backbone in ['34', '18']:
            self.model = resnet(backbone, pretrained=pretrained)
            feature_dim = 512
            self.pool = nn.Conv2d(feature_dim, hidden_dim, 1)

        elif backbone in ['50', '101']:
            self.model = resnet(backbone, pretrained=pretrained)
            feature_dim = 2048
            self.pool = nn.Conv2d(feature_dim, hidden_dim, 1)

        elif backbone in ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']:
            self.model = mobilenet(backbone, pretrained=pretrained)
            feature_dim = 1280
            self.pool = nn.Conv2d(feature_dim, hidden_dim, 1)

        else:
            raise NotImplementedError(backbone)

        # -------------------------------------------------------
        # 🔥 DETR-style Transformer Encoder / Decoder
        # -------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=num_decoder_layers)

        # -------------------------------------------------------
        # 🔥 Learnable queries: one per lane (or rail)
        # -------------------------------------------------------
        self.query_embed = nn.Embedding(self.num_lanes, hidden_dim)

        # -------------------------------------------------------
        # 🔥 FFN that maps each query to a 200×52 map
        # -------------------------------------------------------
        self.output_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_rows * self.num_cols)
        )

    def forward(self, x):
        # Backbone output
        feat = self.model(x)              # (B, C, H', W')

        feat = self.pool(feat)            # (B, hidden, 9, 25) for your size

        B, C, Hf, Wf = feat.shape

        # -----------------------------------------
        # Prepare encoder tokens
        # -----------------------------------------
        src = feat.flatten(2).permute(2, 0, 1)  # (tokens=Hf*Wf, B, hidden)

        memory = self.encoder(src)

        # -----------------------------------------
        # Prepare queries
        # -----------------------------------------
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        # (num_lanes, B, hidden)

        # -----------------------------------------
        # Transformer decoder
        # -----------------------------------------
        dec_out = self.decoder(queries, memory)  # (num_lanes, B, hidden)

        # -----------------------------------------
        # FFN → row-classification map
        # -----------------------------------------
        out = self.output_ffn(dec_out)           # (num_lanes, B, 200*52)
        out = out.permute(1, 0, 2)               # (B, lanes, flat)

        out = out.view(B, self.num_lanes,
                       self.num_rows, self.num_cols)

        # return (B, rows, cols, lanes)
        return out.permute(0, 2, 3, 1)
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


