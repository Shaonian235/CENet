import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.nn import functional as F
from lib.Swin_V2 import SwinTransformerV2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, dim, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        # print(f'c1:{c1.shape}')
        c2 = self.layer1(c1)
        # print(f'c2:{c2.shape}')

        c3 = self.layer2(c2)
        # print(f'c3:{c3.shape}')
        c4 = self.layer3(c3)
        # print(f'c4:{c4.shape}')
        c5 = self.layer4(c4)
        # print(f'c5:{c5.shape}')

        # Top-down
        p5 = self.toplayer(c5)
        # print(f'p5:{p5.shape}')
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        # print(f'latlayer1(c4):{self.latlayer1(c4).shape}, p4:{p4.shape}')

        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # print(f'latlayer1(c3):{self.latlayer2(c3).shape}, p3:{p3.shape}')

        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # print(f'latlayer1(c2):{self.latlayer3(c2).shape}, p2:{p2.shape}')

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


def FPN101(dim):
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(dim, Bottleneck, [2, 2, 2, 2])


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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, max_tokens: int = 256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_tokens = max_tokens  # 超过就降采样 K/V

        head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        # projection layers
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_feat: (B, C, H, W)  -> used as Query
            kv_feat: (B, C, H, W) -> used as Key/Value
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = q_feat.shape
        N = H * W

        # flatten
        # --- 关键优化：只下采样 K/V ---
        kv = kv_feat
        if N > self.max_tokens:
            # 最简单：2x 平均池化，N 减到 1/4
            kv = F.avg_pool2d(kv, kernel_size=2, stride=2, ceil_mode=True)
        Hk, Wk = kv.shape[2], kv.shape[3]
        Nk = Hk * Wk

        q = q_feat.permute(0, 2, 3, 1).reshape(B, N, C)
        kv = kv.permute(0, 2, 3, 1).reshape(B, Nk, C)

        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # project
        # project (注意 k/v 使用 Nk)
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, heads, N, head_dim)
        k = self.k_proj(kv).reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1,
                                                                                          2)  # (B, heads, Nk, head_dim)
        v = self.v_proj(kv).reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1,
                                                                                          2)  # (B, heads, Nk, head_dim)

        # attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, heads, N, Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        # reshape back
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out


# Information fusion module
class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.pre_r = nn.Conv2d(dim, dim, 1)
        self.pre_d = nn.Conv2d(dim, dim, 1)

        self.sum_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),  # 保持尺寸不变
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),  # 1x1卷积进一步处理
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.cross_attention_r = CrossAttention(dim, num_heads=8, qkv_bias=True)
        self.cross_attention_d = CrossAttention(dim, num_heads=8, qkv_bias=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = nn.BatchNorm2d(dim)
        self.sum = nn.Sequential(nn.Conv2d(dim , dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.PReLU())
        self.last = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3,padding=1), nn.BatchNorm2d(dim), nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3,padding=1), nn.BatchNorm2d(dim), nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.PReLU())

    def forward(self, r, d):
        r1 = self.pre_r(r)
        d1 = self.pre_d(d)
        sum = r + d

        sum_branch = self.sum_branch(sum)
        r1 = self.cross_attention_r(r1, d1)
        d1 = self.cross_attention_d(d1, r1)
        r1 = self.bn1(self.conv1(r1))
        d1 = self.bn1(self.conv1(d1))
        Attr = self.avg(r1)
        Attd = self.avg(d1)
        r1 = Attr * d1
        d1 = Attd * r1

        F_c = torch.cat((r1, d1), dim=1)
        F_c = self.last(F_c)
        F_c = F_c+sum_branch
        out = self.sum(F_c)



        return out


###############################################################################
class SCA_ConvBlock(nn.Module):
    """
    Lite版：GAP + 1x1 + Sigmoid 的通道注意力（可选ECA）
    直接替换你原来的 SCA_ConvBlock，接口不变。
    """

    def __init__(self, c, eca_ks=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # ECA 版（1D卷积建模局部跨通道交互）, 或者改成 nn.Sequential(nn.Conv2d(c, c//r,1), nn.ReLU(), nn.Conv2d(c//r, c,1))
        self.eca = nn.Conv1d(1, 1, kernel_size=eca_ks, padding=eca_ks // 2, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.gap(x)  # (B,C,1,1)
        y = y.view(b, 1, c)  # (B,1,C)
        y = self.eca(y).view(b, c, 1, 1)  # (B,C,1,1)
        y = torch.sigmoid(y)
        out = x * y + x  # 残差
        return self.bn(out)


class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿宽度方向池化，得到 H × 1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿高度方向池化，得到 1 × W

        mip = max(8, inp // reduction)  # 中间维度，防止太小

        # 共享 1x1 卷积
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        # 两个方向的卷积
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 沿宽度方向池化 (H × 1)
        x_h = self.pool_h(x)  # [N, C, H, 1]

        # 沿高度方向池化 (1 × W)，注意要转置回来
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1] → [N, C, 1, W]

        # 拼接
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]

        # 共享卷积
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 再分开
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [N, C, 1, W]

        # 两个方向的注意力
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 融合注意力
        out = identity * a_h * a_w
        return out


class EdgeAware(nn.Module):
    def __init__(self, dim1, dim2):
        super(EdgeAware, self).__init__()

        self.conv1 = nn.Conv2d(dim1, dim1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim2, dim1, kernel_size=3, padding=1)
        # self.conv_sum =nn.Conv2d(dim1, dim1, kernel_size=3,padding=1)
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.avg2 = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.sca = SCA_ConvBlock(dim1)
        # self.down =nn.MaxPool2d(2, 2)
        self.last = nn.Conv2d(dim1, dim1, kernel_size=3, padding=1)
        self.coosa = CoordAttention(dim1, dim1)

    def forward(self, f1, f2):
        _, _, h, w = f1.shape
        f2_up = F.interpolate(f2, (h, w))

        f2_up = self.conv2(f2_up)
        f1 = self.conv1(f1)

        max_pool1 = self.max_pool1(f1)
        avg_poo1 = self.avg2(f1)
        a = max_pool1 * avg_poo1
        max_pool2 = self.max_pool2(f2_up)  # 最大池化 (Max Pool) → 更关注显著的局部强响应，比如目标边缘、纹理突出的区域。
        avg_pool2 = self.avg2(f2_up)  # 平均池化 (Avg Pool) → 更关注整体的全局分布，比如背景与目标的均衡信息。
        b = max_pool2 * avg_pool2
        # f_sum = f1 + f2_up
        # f_sum = self.sca(self.down(self.conv_sum(f_sum)))
        sum = a + b
        edge_yin_zi = torch.sigmoid(self.last(sum))
        last = sum * edge_yin_zi + sum
        last = self.coosa(last)
        return last





class GLFD(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GLFD, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
        )

        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 1)
        self.conv_cat1 = BasicConv2d(2 * out_channel, out_channel, 1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.bran1 = BasicConv2d(in_channel, in_channel, 3, padding=1)

    def forward(self, x):
        m = self.bran1(x)

        x0 = self.branch0(m)
        x1 = self.branch1(m)
        x2 = self.branch2(m)

        xl = self.local_att(m)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        xg = (x_cat + self.conv_res(m))
        xl = (xl + self.conv_res(m))

        xxl0 = xg + xl
        xxl1 = xg * xl
        xg0 = self.conv_cat1(torch.cat((xxl0, xxl1), 1))
        x = xg0 + self.conv_res(m)
        return x

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # Backbone model
        self.swin_rgb = SwinTransformerV2()
        self.swin_depth = SwinTransformerV2()
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)
        self.fpn_r_128 = FPN101(128)
        self.fpn_r_256 = FPN101(256)
        self.fpn_d_128 = FPN101(128)
        self.fpn_d_256 = FPN101(256)
        self.fpn_i  = nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1)
        self.fpn_j  = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        ####################################################
        ## new change channle
        ####################################################
        self.fu_1c = nn.Conv2d(256, 128, 1)
        self.fu_2c = nn.Conv2d(512, 256, 1)
        self.fu_3c = nn.Conv2d(1024, 512, 1)
        self.fu_4c = nn.Conv2d(2048, 1024, 1)

        ###############################################
        # information fusion
        ###############################################
        self.fu0 = Fusion(128)
        self.fu1 = Fusion(256)
        self.fu2 = Fusion(512)
        self.fu3 = Fusion(1024)
        ###################################################
        # Edge Aware
        ###################################################
        self.edge = EdgeAware(128, 256)
        self.conv_1024 = nn.Conv2d(1024, 128, kernel_size=1)
        self.conv_512 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_512_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_512_2 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_512_3 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_edge = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_edge1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_edge2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_edge3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        ###############################################
        # decoders fusion
        ###############################################
        self.glfd =GLFD(1024, 512)
        self.glfd1 =GLFD(1024, 256)
        self.glfd2 =GLFD(512+128, 128)
        self.glfd3 =GLFD(256, 128)
        self.out = nn.Conv2d(128, 1, kernel_size=1)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.super1 = nn.Conv2d(128,1,1)
        self.super2 = nn.Conv2d(256,1,1)
        self.super3 = nn.Conv2d(512,1,1)

        if self.training:
            self.initialize_weights()

    def forward(self, imgs, depths):

        stage_rgb = self.swin_rgb(imgs)
        stage_depth = self.swin_depth(self.layer_dep0(depths))
        # dep_0, dep_1, dep_2, dep_3, dep_4 = self.layer_dep(self.layer_dep0(depths))

        img_0 = stage_rgb[0]  # [b,128,64,64]

        img_1 = stage_rgb[1]  # [b,256,32,32]
        img_2 = stage_rgb[2]  # [b,512,16,16]
        # img_3 = stage_rgb[3]         #[b,1024,8,8]
        img_4 = stage_rgb[4]  # [b,1024,8,8]

        dep_0 = stage_depth[0]
        dep_1 = stage_depth[1]
        dep_2 = stage_depth[2]
        dep_4 = stage_depth[4]

        ###########################################
        # FPN
        # torch.Size([2, 256, 16, 16])
        # torch.Size([2, 256, 8, 8])
        # torch.Size([2, 256, 4, 4])
        # torch.Size([2, 256, 2, 2])
        ###########################################

        r_00, r_01, r_02, r_03 = self.fpn_r_128(img_0)
        print(r_00.shape, r_01.shape, r_02.shape, r_03.shape)
        r_10, r_11, r_12, r_13 = self.fpn_r_256(img_1)
        dep_00, dep_01, dep_02, dep_03 = self.fpn_d_128(dep_0)
        dep_10, dep_11, dep_12, dep_13 = self.fpn_d_256(dep_1)

        r_03 = F.interpolate(r_03, size=(16, 16), mode='bilinear')
        r_02 = F.interpolate(r_02, size=(16, 16), mode='bilinear')
        r_01 = F.interpolate(r_01, size=(16, 16), mode='bilinear')

        r_13 = F.interpolate(r_13, size=(8, 8), mode='bilinear')
        r_12 = F.interpolate(r_12, size=(8, 8), mode='bilinear')
        r_11 = F.interpolate(r_11, size=(8, 8), mode='bilinear')

        dep_03 = F.interpolate(dep_03, size=(16, 16), mode='bilinear')
        dep_02 = F.interpolate(dep_02, size=(16, 16), mode='bilinear')
        dep_01 = F.interpolate(dep_01, size=(16, 16), mode='bilinear')

        dep_13 = F.interpolate(dep_13, size=(8, 8), mode='bilinear')
        dep_12 = F.interpolate(dep_12, size=(8, 8), mode='bilinear')
        dep_11 = F.interpolate(dep_11, size=(8, 8), mode='bilinear')

        d0 = torch.cat((dep_00, dep_01, dep_02, dep_03), dim=1)  # 1024 16
        d1 = torch.cat((dep_10, dep_11, dep_12, dep_13), dim=1)  # 1024 8
        r0 = torch.cat((r_00, r_01, r_02, r_03), dim=1)
        r1 = torch.cat((r_10, r_11, r_12, r_13), dim=1)
        d0 = self.fpn_i(d0)#1024->128
        d1 = self.fpn_j(d1)# 1024->256
        r0 = self.fpn_i(r0)
        r1 = self.fpn_j(r1)
        ###########################################################
        # Fusion
        ###########################################################
        fu0 = self.fu0(r0, d0)  # 128 16
        print(fu0.shape)
        print("11111")
        fu1 = self.fu1(r1, d1)  # 256 8
        fu3 = self.fu2(img_2, dep_2)  # 512 16
        fu4 = self.fu3(img_4, dep_4)  # 1024 ,8


        ##############################################################
        # Edge
        ##############################################################
        edge1 = self.edge(img_0, img_1)  # 128,32,32
        _, _, h, w = edge1.size()
        fuu0 = F.interpolate(fu0, scale_factor=4, mode='bilinear', align_corners=False)
        fuu1 = F.interpolate(fu1, scale_factor=4, mode='bilinear', align_corners=False)
        fuu3 = fu3 #512 16
        fuu4 = fu4 #1024 8

        ###############################################
        # Decoder
        ###############################################
        F4 = self.glfd(fuu4) #512
        F4 = F.interpolate(F4,scale_factor=2, mode='bilinear', align_corners=False)
        F43 = torch.cat([F4,fuu3],dim=1)
        F43 = self.glfd1(F43)# 256
        F43 = F.interpolate(F43, scale_factor=2, mode='bilinear', align_corners=False)
        F432 =  torch.cat([F43,fuu1,edge1],dim=1)
        F432 = self.glfd2(F432) #128
        F432 = F.interpolate(F432, scale_factor=2, mode='bilinear', align_corners=False)
        F4321 = torch.cat([F432,fuu0],dim=1)
        F4321 = self.glfd3(F4321)#128
        f3 = F.interpolate(self.super3(F4),scale_factor=16,mode='bilinear', align_corners=False)
        f2 = F.interpolate(self.super2(F43),scale_factor=8,mode='bilinear', align_corners=False)
        f1 = F.interpolate(self.super1(F432),scale_factor=4,mode='bilinear', align_corners=False)
        # torch.Size([2, 1, 16, 16]) torch.Size([2, 1, 32, 32]) torch.Size([2, 1, 64, 64])
        print(f3.shape,f2.shape,f1.shape)

        out1 = self.upsample_4(self.out(F4321))

        return out1,f3,f2,f1

    def initialize_weights(self):  # 加载预训练模型权重，做初始化
        # rgb_path = '/root/lanyun-tmp/Net/ADINet/lib/swinv2_base_patch4_window16_256.pth'
        rgb_path = 'E:/code/ADI/MyNet/ADINet/lib/swinv2_base_patch4_window16_256.pth'

        # rgb_path = '/root/lanyun-tmp/Net/ADINet/lib/swinv2_base_patch4_window16_256.pth'
        if os.path.exists(rgb_path):
            self.swin_rgb.load_state_dict(torch.load(rgb_path)['model'], strict=False)
            print("RGB Swin weights loaded successfully")
        else:
            print(f"Warning: RGB pretrained weights not found at {rgb_path}")
        if os.path.exists(rgb_path):
            self.swin_depth.load_state_dict(torch.load(rgb_path)['model'], strict=False)
            print("Depth Swin weights loaded successfully")


if __name__ == '__main__':
    from thop import profile

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rgb = torch.rand((2, 3, 256, 256)).cuda()
    depth = torch.rand((2, 1, 256, 256)).cuda()
    model = MyNet().cuda()
    rgb.to(device)
    depth.to(device)
    model.to(device)
    flops1, params1 = profile(model, inputs=(rgb, depth))
    #
    print('params:%.2f(M)' % ((params1) / 1000000))
    print('flops:%.2f(G)' % ((flops1) / 1000000000))
    l = model(rgb, depth)