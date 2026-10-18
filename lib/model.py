import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


###################################################################
class MFAM(nn.Module):
    """Multi-scale Feature Aggregation Module"""

    def __init__(self, in_channels, out_channels):
        super(MFAM, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)

        self.conv_3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_5_1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x1    = self.conv_1_1(x)
        x2    = self.conv_1_2(x)
        x3    = self.conv_1_3(x)

        x_3_1 = self.relu(self.conv_3_1(x2))
        x_5_1 = self.relu(self.conv_5_1(x3))

        x_3_2 = self.relu(self.conv_3_2(x_3_1 + x_5_1))
        x_5_2 = self.relu(self.conv_5_2(x_5_1 + x_3_1))

        x_mul = torch.mul(x_3_2, x_5_2)
        out   = self.relu(x1 + self.conv_1_5(x_mul + x_3_1 + x_5_1))

        return out


###################################################################
class FeaFusion(nn.Module):
    """Cross-level Feature Fusion (part of CFPM)"""
    def __init__(self, channels):
        self.init__ = super(FeaFusion, self).__init__()

        self.relu     = nn.ReLU()
        self.layer1   = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        self.layer2_1 = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1)
        self.layer2_2 = nn.Conv2d(channels, channels // 4, kernel_size=3, stride=1, padding=1)

        self.layer_fu = nn.Conv2d(channels // 4, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        wweight    = nn.Sigmoid()(self.layer1(x1 + x2))

        xw_resid_1 = x1 + x1.mul(wweight)
        xw_resid_2 = x2 + x2.mul(wweight)

        x1_2       = self.layer2_1(xw_resid_1)
        x2_2       = self.layer2_2(xw_resid_2)

        out        = self.relu(self.layer_fu(x1_2 + x2_2))

        return out


###################################################################
class FeaProp(nn.Module):
    """Feature Propagation via gated attention (part of CFPM)"""
    def __init__(self, in_planes):
        self.init__ = super(FeaProp, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer_1  = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_planes), act_fn)
        self.layer_2  = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(in_planes), act_fn)

        self.gate_1   = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)
        self.gate_2   = nn.Conv2d(in_planes * 2, 1, kernel_size=1, bias=True)

        self.softmax  = nn.Softmax(dim=1)

    def forward(self, x10, x20):
        x1 = self.layer_1(x10)
        x2 = self.layer_2(x20)

        cat_fea = torch.cat([x1, x2], dim=1)

        att_vec_1 = self.gate_1(cat_fea)
        att_vec_2 = self.gate_2(cat_fea)

        att_vec_cat  = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)

        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2

        return x_fusion


###################################################################
class FAPNet(nn.Module):
    """
    Feature Aggregation and Propagation Network for Camouflaged Object Detection.
    Backbone: ConvNeXt-Tiny (torchvision)

    ConvNeXt-Tiny feature map channels at each stage:
        Stage 1  (features[1]):  96  channels  [3 blocks]
        Stage 3  (features[3]): 192  channels  [3 blocks]
        Stage 5  (features[5]): 384  channels  [9 blocks]
        Stage 7  (features[7]): 768  channels  [3 blocks]

    Opt-in experiment flags (defaults reproduce the original baseline exactly,
    including state_dict keys, so old checkpoints load with strict=True):

        boundary_head : 'single'     -> original boundary branch (control)
                        'multiscale' -> H2: fuses a deeper neighbouring feature
                                        (stage-2, 192 ch, H/8) into the shallow
                                        (stage-1, H/4) boundary feature
        uncertainty   : False -> original 5-tuple output
                        True  -> H3: adds a spatial uncertainty head

    Output contract:
        default         : (out5, out4, out3, out2, edge_out)
        uncertainty=True: (out5, out4, out3, out2, edge_out, logvar)
    out2 (index 3) is always the final segmentation logit map.
    `logvar` is a bounded per-pixel log-variance in [-unc_max, unc_max] at full
    resolution (larger = more uncertain).
    """

    def __init__(self, channel=64, opt=None, boundary_head='single',
                 uncertainty=False, unc_max=2.0):
        super(FAPNet, self).__init__()
        assert boundary_head in ('single', 'multiscale'), boundary_head

        act_fn   = nn.ReLU(inplace=True)
        self.nf  = channel
        self.boundary_head = boundary_head
        self.uncertainty   = uncertainty
        self.unc_max       = unc_max

        # ---- Backbone: ConvNeXt-Tiny (random init; checkpoints supply weights) ----
        self.backbone = models.convnext_tiny(weights=None)

        # ---- MFAM modules ----
        self.rf1  = MFAM(96,  self.nf)
        self.rf2  = MFAM(96,  self.nf)
        self.rf3  = MFAM(192, self.nf)
        self.rf4  = MFAM(384, self.nf)
        self.rf5  = MFAM(768, self.nf)

        # ---- Cross-level Fusion (FeaFusion) ----
        self.cfusion2 = FeaFusion(self.nf)
        self.cfusion3 = FeaFusion(self.nf)
        self.cfusion4 = FeaFusion(self.nf)
        self.cfusion5 = FeaFusion(self.nf)

        # ---- Feature Propagation (FeaProp / gate) ----
        self.cgate5 = FeaProp(self.nf)
        self.cgate4 = FeaProp(self.nf)
        self.cgate3 = FeaProp(self.nf)
        self.cgate2 = FeaProp(self.nf)

        # ---- Decoder convs ----
        self.de_5 = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.de_4 = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.de_3 = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.de_2 = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)

        # ---- Boundary Guidance Module (BGM) ----
        self.edge_conv0 = nn.Sequential(nn.Conv2d(96,      self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv1 = nn.Sequential(nn.Conv2d(96,      self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv2 = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv3 = BasicConv2d(self.nf, 1, kernel_size=3, padding=1)

        # ---- H2: optional multi-scale boundary head (created only when requested) ----
        if self.boundary_head == 'multiscale':
            self.edge_ms_proj = nn.Sequential(nn.Conv2d(192, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
            self.edge_ms_fuse = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)

        # ---- Fusion of edge guidance into decoder ----
        self.fu_5 = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.fu_4 = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.fu_3 = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.fu_2 = nn.Sequential(nn.Conv2d(self.nf * 2, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)

        # ---- Output heads ----
        self.layer_out5 = nn.Sequential(nn.Conv2d(self.nf, 1, kernel_size=3, stride=1, padding=1))
        self.layer_out4 = nn.Sequential(nn.Conv2d(self.nf, 1, kernel_size=3, stride=1, padding=1))
        self.layer_out3 = nn.Sequential(nn.Conv2d(self.nf, 1, kernel_size=3, stride=1, padding=1))
        self.layer_out2 = nn.Sequential(nn.Conv2d(self.nf, 1, kernel_size=3, stride=1, padding=1))

        # ---- H3: optional spatial uncertainty head ----
        if self.uncertainty:
            self.unc_head = nn.Sequential(
                nn.Conv2d(self.nf, self.nf // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.nf // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.nf // 2, 1, kernel_size=3, stride=1, padding=1))
            # zero-init the last conv: logvar starts at 0, so the weighted loss
            # starts identical to the unweighted one
            nn.init.zeros_(self.unc_head[-1].weight)
            nn.init.zeros_(self.unc_head[-1].bias)

        # ---- Upsamplers ----
        self.up_2  = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,  mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,  mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, xx):
        # ---- ConvNeXt-Tiny feature extraction ----
        f = self.backbone.features
        x0 = f[0](xx)          # stem          (BS, 96,  H/4,  W/4)
        x1 = f[1](x0)          # stage-1       (BS, 96,  H/4,  W/4)
        x2 = f[2](x1)          # downsample    (BS, 192, H/8,  W/8)
        x3 = f[3](x2)          # stage-2       (BS, 192, H/8,  W/8)
        x4 = f[4](x3)          # downsample    (BS, 384, H/16, W/16)
        x5 = f[5](x4)          # stage-3       (BS, 384, H/16, W/16)
        x6 = f[6](x5)          # downsample    (BS, 768, H/32, W/32)
        x7 = f[7](x6)          # stage-4       (BS, 768, H/32, W/32)

        # ---- MFAM ----
        xf1 = self.rf1(x1)
        xf2 = self.rf2(x1)
        xf3 = self.rf3(x3)
        xf4 = self.rf4(x5)
        xf5 = self.rf5(x7)

        # ---- BGM: Boundary Guidance Module ----
        x21           = self.edge_conv1(x1)                          # (BS, 64, H/4, W/4)
        edge_guidance = self.edge_conv2(self.edge_conv0(x1) + x21)   # (BS, 64, H/4, W/4)
        if self.boundary_head == 'multiscale':
            # H2: bring the deeper neighbouring feature (H/8) up to H/4 and fuse
            deep = self.edge_ms_proj(x3)
            deep = F.interpolate(deep, size=edge_guidance.shape[2:], mode='bilinear', align_corners=True)
            edge_guidance = self.edge_ms_fuse(torch.cat((edge_guidance, deep), dim=1))
        edge_out      = self.up_4(self.edge_conv3(edge_guidance))     # (BS, 1,  H,   W)

        # ---- Decoder with CFPM (FeaFusion + FeaProp) ----
        ### Layer 5  (H/16)
        en_fusion5   = self.cfusion5(self.up_2(xf5), xf4)
        out_gate_fu5 = self.fu_5(torch.cat((en_fusion5, F.interpolate(edge_guidance, scale_factor=1/4, mode='bilinear')), dim=1))
        out5         = self.up_16(self.layer_out5(out_gate_fu5))

        ### Layer 4  (H/8)
        de_feature4  = self.de_4(self.up_2(en_fusion5))
        en_fusion4   = self.cfusion4(self.up_2(xf4), xf3)
        out_gate4    = self.cgate4(en_fusion4, de_feature4)
        out_gate_fu4 = self.fu_4(torch.cat((out_gate4, F.interpolate(edge_guidance, scale_factor=1/2, mode='bilinear')), dim=1))
        out4         = self.up_8(self.layer_out4(out_gate_fu4))

        ### Layer 3  (H/4)
        de_feature3  = self.de_3(self.up_2(out_gate4))
        en_fusion3   = self.cfusion3(self.up_2(xf3), xf2)
        out_gate3    = self.cgate3(en_fusion3, de_feature3)
        out_gate_fu3 = self.fu_3(torch.cat((out_gate3, edge_guidance), dim=1))
        out3         = self.up_4(self.layer_out3(out_gate_fu3))

        ### Layer 2  (H/2 -> H)
        de_feature2  = self.de_2(self.up_2(out_gate3))
        en_fusion2   = self.cfusion2(self.up_2(xf2), self.up_2(xf1))
        out_gate2    = self.cgate2(en_fusion2, de_feature2)
        out_gate_fu2 = self.fu_2(torch.cat((out_gate2, self.up_2(edge_guidance)), dim=1))
        out2         = self.up_2(self.layer_out2(out_gate_fu2))

        if self.uncertainty:
            raw    = self.up_2(self.unc_head(out_gate_fu2))                 # (BS, 1, H, W)
            logvar = self.unc_max * torch.tanh(raw / self.unc_max)          # bounded
            return out5, out4, out3, out2, edge_out, logvar

        return out5, out4, out3, out2, edge_out


###################################################################
def load_fapnet(path, device, boundary_head=None, uncertainty=None, strict=True, channel=64):
    """
    Build FAPNet and load a checkpoint (raw state_dict or full training checkpoint).

    boundary_head / uncertainty: pass None to infer from the 'config' stored in
    checkpoints written by the updated train.py (falls back to the baseline:
    'single' / False for old checkpoints).

    Returns (model, checkpoint_dict_or_empty, (missing_keys, unexpected_keys)).
    """
    ckpt    = torch.load(path, map_location=device)
    is_full = isinstance(ckpt, dict) and 'model_state_dict' in ckpt
    state   = ckpt['model_state_dict'] if is_full else ckpt
    cfg     = ckpt.get('config', {}) if is_full else {}

    if boundary_head is None:
        boundary_head = cfg.get('boundary_head', 'single')
    if uncertainty is None:
        uncertainty = cfg.get('uncertainty', 'none') != 'none'

    model  = FAPNet(channel=channel, boundary_head=boundary_head, uncertainty=bool(uncertainty)).to(device)
    result = model.load_state_dict(state, strict=strict)
    model.eval()
    return model, (ckpt if is_full else {}), (list(result.missing_keys), list(result.unexpected_keys))
