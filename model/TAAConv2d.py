import torch
import torch.nn as nn
from torch.autograd import Variable
from model.SpatioTemporalAttention import SpatioTemporalAttention

class TAAConv2d(nn.Module):
    '''
    Temporal Attention Augmented Convolutional defined in "Attention Augmented ConvLSTM for Environment Prediction"

    # Arguments:
        input_channels: number of channels in the input
        output_channels: number of channels in the hidden representation
        kernel_size: filter size used in the convolution operator
        padding: padding
        num_past_frames: attention horizon
        dk: ratio of number of channels in the key/query to number of channels in the output
        dv: ratio of number of channels in the value to number of channels in the output
        Nh: number of heads
        positional_encoding: whether to add positional encoding in the attention calculation
        forget_bias: whether to add forget bias when training in the forget gate

    '''
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

    def forward(self, rep, history):

        attn_out = self.spatio_temporal_attention(rep, history)

        if self.dv != self.out_channels:
            conv_out = self.conv(rep)
            concat_out = torch.cat((conv_out, attn_out), dim=1)
        else:
            concat_out = attn_out

        return concat_out
