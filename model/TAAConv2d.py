import torch
import torch.nn as nn
from torch.autograd import Variable
from model.SpatioTemporalAttention import SpatioTemporalAttention

class TAAConv2d(nn.Module):

    """
        Describe temporal self attention

        The same varaible names based on "Attention Augmneted Convolutional Networks".

        https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/in_paper_attention_augmented_conv/attention_augmented_conv.pyxs

    """
    def __init__(self, input_channels, output_channels, kernel_size, padding, num_past_frames, dk, dv, Nh, width, height, attention_input_mode, relative = True):

        super(TAAConv2d, self).__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_past_frames = num_past_frames
        self.dk = dk * self.out_channels//1000
        self.dv = dv * self.out_channels//1000
        self.Nh = Nh
        self.width = width
        self.height = height
        self.attention_input_mode = attention_input_mode
        self.Nh = Nh
        self.relative = relative

        if self.dv != self.out_channels:
            self.conv = nn.Conv2d(self.in_channels, self.out_channels- self.dv, self.kernel_size,1, self.padding, bias=False)

        self.spatio_temporal_attention = SpatioTemporalAttention(self.out_channels, self.num_past_frames, self.kernel_size, self.padding,
                                                                    self.dk, self.dv, self.Nh, self.width, self.height, self.relative)

        #MODIFIED
        #self.output = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, rep, history):

        attn_out = self.spatio_temporal_attention(rep, history)

        if self.dv != self.out_channels:
            conv_out = self.conv(rep)
            concat_out = torch.cat((conv_out, attn_out), dim=1)
        else:
            concat_out = attn_out

        #MODIFIED
        #out = self.output(concat_out)

        return concat_out
