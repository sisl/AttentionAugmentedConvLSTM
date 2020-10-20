import torch
import torch.nn as nn
from torch.autograd import Variable
from model.SpatialAttention import SpatialAttention


class SpatioTemporalAttention(nn.Module):

    '''
        Temporal Attention used in "Attention Augmented ConvLSTM for Environment Prediction".
        Evaluate attention for each pair (current representation, past representation).

        # Arguments:
            input_channels: number of channels in the input
            kernel_size: filter size used in the convolution operator
            num_past_frames: attention horizon
            dk: number of channels in the key/query
            dv: number of channels in the value
            relative: whether to add positional encoding in the attention calculation
            forget_bias: whether to add forget bias when training in the forget gate
            padding: padding
    '''

    def __init__(self, input_channels,  num_past_frames, kernel_size, padding,  dk, dv, Nh, width, height, relative = True):

        super(SpatioTemporalAttention, self).__init__()
        self.in_channels = input_channels
        self.num_past_frames = num_past_frames
        self.kernel_size = kernel_size
        self.padding = padding
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.width = width
        self.height = height
        self.Nh = Nh
        self.relative = relative
        self.spatial_attn_out = SpatialAttention(self.in_channels, self.kernel_size, self.padding, self.dk, self.dv, self.Nh, self.width, self.height, self.relative)
        self.temporal_weights = nn.Conv2d(self.num_past_frames,1,kernel_size=1, stride=1,bias=False)

    def forward(self, inputs, history):

        batch_size, channels, height, width = inputs.size()
        _, timesteps, channels_history,_,_ = history.size()

        # SpatialAttention perfroms regular attention for each (inputs, history[t]) pair
        history = torch.reshape(history, (batch_size*timesteps, channels_history, height, width))
        repeating_inputs = torch.repeat_interleave(inputs, repeats = timesteps, dim=0)

        #Considered both inputs and history as keys.
        spatial_attn_out = self.spatial_attn_out(repeating_inputs, history, history) #querry, keys, values

        #Output should be batch_size*timesteps, self.dv, heigh, width
        spatial_attn_out = torch.reshape(spatial_attn_out, (batch_size, timesteps, self.dv, height, width))
        spatial_attn_out = torch.transpose(spatial_attn_out, 1,2)
        spatial_attn_out = torch.reshape(spatial_attn_out, (batch_size*self.dv, timesteps, height, width))
        out = self.temporal_weights(spatial_attn_out).squeeze(1)
        out = torch.reshape(out, (batch_size, self.dv, height, width))

        return out
