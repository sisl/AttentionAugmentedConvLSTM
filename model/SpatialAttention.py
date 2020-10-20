import torch
import torch.nn as nn
from torch.autograd import Variable
import math
#import GPUtil

class SpatialAttention(nn.Module):

    """
    Spatial Attention implementation.
    AAConv2d with modified query, key and values.
    The same varaible names based on "Attention Augmneted Convolutional Networks".

    Based on:
        -Bello, Irwan, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V. Le. Official
        Tensorflow implementation of Attention Augmented Conv2d "Attention augmented convolutional networks."
        In Proceedings of the IEEE International Conference on Computer Vision, pp. 3286-3295. 2019. Code included in the paper.
        -https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/in_paper_attention_augmented_conv/attention_augmented_conv.pyxs

    """
    def __init__(self, input_channels, kernel_size, padding, dk, dv, Nh, width, height, relative = True):

        super(SpatialAttention, self).__init__()
        self.in_channels = input_channels
        self.kernel_size = 1
        self.padding = 0
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.width = width
        self.height = height
        self.Nh = Nh
        self.relative = relative

        self.q_conv = nn.Conv2d(self.in_channels, self.Nh*self.dk, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        self.k_conv = nn.Conv2d(self.in_channels, self.Nh*self.dk, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        self.v_conv = nn.Conv2d(self.in_channels, self.Nh*self.dv, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        self.attn_out = nn.Conv2d(self.Nh*self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.width - 1, self.dk), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.height - 1, self.dk), requires_grad=True))

    def forward(self, q_x, k_x, v_x):

        batch, channels, height, width = q_x.size()
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(q_x, k_x, v_x)
        logits = torch.matmul(flat_q.transpose(2,3), flat_k)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits_bello(q)
            logits += h_rel_logits
            logits += w_rel_logits

        weights = torch.nn.functional.softmax(logits, dim=-1)
        spatial_attn_out = torch.matmul(weights, flat_v.transpose(2,3))
        spatial_attn_out = torch.reshape(spatial_attn_out, (batch, self.Nh, self.dv, height, width))
        attn_out = self.combine_heads_2d(spatial_attn_out)
        attn_out = self.attn_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, q_x, k_x, v_x):

        N, _, H, W = q_x.size()
        q = self.q_conv(q_x)
        k = self.k_conv(k_x)
        v = self.v_conv(v_x)

        q = self.split_heads_2d(q, self.Nh)
        k = self.split_heads_2d(k, self.Nh)
        v = self.split_heads_2d(v, self.Nh)

        dkh = self.dk
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, self.Nh, self.dk, H * W))
        flat_k = torch.reshape(k, (N, self.Nh, self.dk, H * W))
        flat_v = torch.reshape(v, (N, self.Nh, self.dv, H * W))

        return flat_q, flat_k, flat_v, q, k, v

    def relative_logits_bello(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).cuda()
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).cuda()
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def reset_parameters(self):
        self.q_conv.reset_parameters()
        self.k_conv.reset_parameters()
        self.v_conv.reset_parameters()
