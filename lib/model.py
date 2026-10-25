import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


MFAM_VARIANTS = ('original', 'depthwise', 'dilated', 'factorized', 'combined')


def validate_mfam_variant(variant):
    """Raise a clear ValueError if `variant` is not a known MFAM variant."""
    if variant not in MFAM_VARIANTS:
        raise ValueError(
            "Unknown MFAM variant '{}'. Valid choices: {}.".format(variant, ', '.join(MFAM_VARIANTS)))
    return variant


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
    """Multi-scale Feature Aggregation Module

    Supports five MFAM_VARIANTS that swap only the four spatial
    branch convolutions (conv_3_1, conv_3_2, conv_5_1, conv_5_2).
    The branch/fusion topology, the post-fusion conv_1_5, the 1x1
    projection convolutions (conv_1_{1-4}), the feature dimensions,
    and the five-output contract of FAPNet are all unchanged.

    variant='original' (default) reproduces the existing architecture
    and state-dict layout exactly so current checkpoints load without
    any key renaming.
    """

    def __init__(self, in_channels, out_channels, variant='original'):
        super(MFAM, self).__init__()
        self.variant = validate_mfam_variant(variant)

        self.relu = nn.ReLU(inplace=True)

        self.conv_1_1 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_2 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_3 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_4 = BasicConv2d(in_channels, out_channels, 1)
        self.conv_1_5 = BasicConv2d(out_channels, out_channels, 3, stride=1, padding=1)

        if variant == 'original':
            self.conv_3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_5_1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
            self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

        elif variant == 'depthwise':
            # Depthwise-separable: depthwise conv (no bias) + 1x1 pointwise (with bias)
            self.conv_3_1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )
            self.conv_3_2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )
            self.conv_5_1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )
            self.conv_5_2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )

        elif variant == 'dilated':
            # 3x3 branches stay standard; 5x5 branches become dilated 3x3 (dilation=2, same receptive field)
            self.conv_3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_5_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
            self.conv_5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)

        elif variant == 'factorized':
            # 3x3 branches stay standard; 5x5 factorized into 1x5 followed by 5x1
            self.conv_3_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_5_1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            )
            self.conv_5_2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
                nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            )

        elif variant == 'combined':
            # 3x3 branches: depthwise-separable 3x3; 5x5 branches: depthwise-separable dilated 3x3
            self.conv_3_1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )
            self.conv_3_2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )
            self.conv_5_1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2,
                          groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )
            self.conv_5_2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2,
                          groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            )

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

    These map to rf1..rf5 as follows (matching the trained checkpoints):
        rf1: 96   (stage-1 — also used for edge branch)
        rf2: 96   (stage-1 again, shared with edge)
        rf3: 192  (stage-3)
        rf4: 384  (stage-5)
        rf5: 768  (stage-7)

    mfam_variant: one of MFAM_VARIANTS. 'original' (default) reproduces the
    existing architecture and state-dict layout exactly.
    """

    def __init__(self, channel=64, opt=None, mfam_variant='original'):
        super(FAPNet, self).__init__()

        act_fn   = nn.ReLU(inplace=True)
        self.nf  = channel
        self.mfam_variant = validate_mfam_variant(mfam_variant)

        # ---- Backbone: ConvNeXt-Base ----
        self.backbone = models.convnext_tiny(weights=None)

        # ---- MFAM modules ----
        # rf1 & rf2 both take the stage-1 output (96 ch)
        # rf3 takes stage-3 (192 ch), rf4 stage-5 (384 ch), rf5 stage-7 (768 ch)
        self.rf1  = MFAM(96,  self.nf, mfam_variant)
        self.rf2  = MFAM(96,  self.nf, mfam_variant)
        self.rf3  = MFAM(192, self.nf, mfam_variant)
        self.rf4  = MFAM(384, self.nf, mfam_variant)
        self.rf5  = MFAM(768, self.nf, mfam_variant)

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
        # edge_conv0 & edge_conv1 both take 96-ch (stage-1) input
        self.edge_conv0 = nn.Sequential(nn.Conv2d(96,      self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv1 = nn.Sequential(nn.Conv2d(96,      self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv2 = nn.Sequential(nn.Conv2d(self.nf, self.nf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(self.nf), act_fn)
        self.edge_conv3 = BasicConv2d(self.nf, 1, kernel_size=3, padding=1)

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

        # ---- Upsamplers ----
        self.up_2  = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=True)
        self.up_4  = nn.Upsample(scale_factor=4,  mode='bilinear', align_corners=True)
        self.up_8  = nn.Upsample(scale_factor=8,  mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, xx):
        # ---- ConvNeXt-Base feature extraction ----
        # features[0]: stem  (downsample x4)  → (BS, 96, H/4, W/4)
        # features[1]: stage-1 blocks (3 blocks) → (BS, 96, H/4, W/4)
        # features[2]: downsample     → (BS, 192, H/8, W/8)
        # features[3]: stage-2 blocks (3 blocks) → (BS, 192, H/8, W/8)
        # features[4]: downsample     → (BS, 384, H/16, W/16)
        # features[5]: stage-3 blocks (9 blocks) → (BS, 384, H/16, W/16)
        # features[6]: downsample     → (BS, 768, H/32, W/32)
        # features[7]: stage-4 blocks (3 blocks) → (BS, 768, H/32, W/32)

        f = self.backbone.features
        x0 = f[0](xx)          # stem          (BS, 96,  H/4,  W/4)
        x1 = f[1](x0)          # stage-1       (BS, 96,  H/4,  W/4)
        x2 = f[2](x1)          # downsample    (BS, 192, H/8,  W/8)
        x3 = f[3](x2)          # stage-2       (BS, 192, H/8,  W/8)
        x4 = f[4](x3)          # downsample    (BS, 384, H/16, W/16)
        x5 = f[5](x4)          # stage-3       (BS, 384, H/16, W/16)
        x6 = f[6](x5)          # downsample    (BS, 768, H/32, W/32)
        x7 = f[7](x6)          # stage-4       (BS, 768, H/32, W/32)

        # ---- MFAM: multi-scale feature aggregation at each level ----
        xf1 = self.rf1(x1)     # (BS, 64, H/4,  W/4)   ← stage-1 (96 ch)
        xf2 = self.rf2(x1)     # (BS, 64, H/4,  W/4)   ← stage-1 again (shared with edge)
        xf3 = self.rf3(x3)     # (BS, 64, H/8,  W/8)   ← stage-2 (192 ch)
        xf4 = self.rf4(x5)     # (BS, 64, H/16, W/16)  ← stage-3 (384 ch)
        xf5 = self.rf5(x7)     # (BS, 64, H/32, W/32)  ← stage-4 (768 ch)

        # ---- BGM: Boundary Guidance Module ----
        # Uses the two lowest-level feature maps (both 96-ch, stage-1)
        x21           = self.edge_conv1(x1)                          # (BS, 64, H/4, W/4)
        edge_guidance = self.edge_conv2(self.edge_conv0(x1) + x21)   # (BS, 64, H/4, W/4)
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

        ### Layer 2  (H/2 → H)
        de_feature2  = self.de_2(self.up_2(out_gate3))
        en_fusion2   = self.cfusion2(self.up_2(xf2), self.up_2(xf1))
        out_gate2    = self.cgate2(en_fusion2, de_feature2)
        out_gate_fu2 = self.fu_2(torch.cat((out_gate2, self.up_2(edge_guidance)), dim=1))
        out2         = self.up_2(self.layer_out2(out_gate_fu2))

        # ---- Output ----
        return out5, out4, out3, out2, edge_out