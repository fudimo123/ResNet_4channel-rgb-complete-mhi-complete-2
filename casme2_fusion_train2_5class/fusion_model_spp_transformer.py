import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNetSPPTokenizer(nn.Module):
    def __init__(self, levels=(1,2,4), pretrained=True):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        # use torchvision resnet18; if you prefer pretrained, load via weights above
        self.conv1, self.bn1, self.relu, self.maxpool = m.conv1, m.bn1, m.relu, m.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.levels = levels
        self.out_channels = 512

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        tokens = []
        for l in self.levels:
            p = F.adaptive_max_pool2d(x, (l, l))  # B,C,l,l
            t = p.flatten(2).transpose(1, 2)      # B,l*l,C
            tokens.append(t)
        return tokens  # list of [B, l*l, C]

class SPPTokenTransformerFusion(nn.Module):
    def __init__(self, num_classes=5, levels=(1,2,4), d_model=384, n_heads=6, num_layers=2, dropout=0.3, pretrained=True, use_cls=True):
        super().__init__()
        self.rgb = ResNetSPPTokenizer(levels, pretrained)
        self.dyn = ResNetSPPTokenizer(levels, pretrained)
        self.proj = nn.Linear(self.rgb.out_channels, d_model)
        self.level_embed = nn.Embedding(len(levels), d_model)
        self.modality_embed = nn.Embedding(2, d_model)
        self.use_cls = use_cls
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def _add_pe(self, t, level_idx, modality_idx):
        B, N, D = t.shape
        device = t.device
        lvl_ids = torch.full((B, N), level_idx, dtype=torch.long, device=device)
        mod_ids = torch.full((B, 1), modality_idx, dtype=torch.long, device=device)
        t = t + self.level_embed(lvl_ids)
        t = t + self.modality_embed(mod_ids).expand(B, N, -1)
        return t

    def forward(self, rgb, dyn):
        rgb_tokens_levels = self.rgb(rgb)
        dyn_tokens_levels = self.dyn(dyn)
        seqs = []
        for i in range(len(rgb_tokens_levels)):
            tr = self.proj(rgb_tokens_levels[i])
            td = self.proj(dyn_tokens_levels[i])
            tr = self._add_pe(tr, i, 0)
            td = self._add_pe(td, i, 1)
            seqs.append(torch.cat([tr, td], dim=1))
        seq = torch.cat(seqs, dim=1)  # B, sum(l*l)*2, D
        if self.use_cls:
            cls = self.cls_token.expand(seq.size(0), -1, -1)
            seq = torch.cat([cls, seq], dim=1)
        out = self.encoder(seq)
        feat = out[:,0] if self.use_cls else out.mean(dim=1)
        return self.head(feat)


def create_fusion_model(num_classes=5, dropout_p=0.5, pretrained=True):
    return SPPTokenTransformerFusion(num_classes=num_classes, dropout=dropout_p, pretrained=pretrained)
