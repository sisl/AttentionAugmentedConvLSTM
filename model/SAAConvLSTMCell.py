import torch
import torch.nn as nn
from torch.autograd import Variable
from model.AAConv2d import AAConv2d

class SAAConvLSTMCell(nn.Module):

    '''
    Self-Attention Augmented Convolutional LSTM defined in "Attention Augmented ConvLSTM for Environment Prediction"

    # Arguments:
        input_channels: number of channels in the input
        hidden_channels: number of channels in the hidden representation
        kernel_size: filter size used in the convolution operator
        num_past_frames: attention horizon
        dk: number of channels in the key/query
        dv: number of channels in the value
        positional_encoding: whether to add positional encoding in the attention calculation
        forget_bias: whether to add forget bias when training in the forget gate
        padding: padding
    '''
    def __init__(self, input_channels, hidden_channels, kernel_size, num_past_frames, dk, dv,
                Nh, width, height, attention_input_mode='representation', positional_encoding = True, forget_bias = 1.0):

        super(SAAConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_past_frames = num_past_frames
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.width = width
        self.height = height
        self.attention_input_mode = attention_input_mode
        self.positional_encoding = positional_encoding
        self.forget_bias = forget_bias
        self.padding = int((kernel_size - 1) / 2)

        self.W_xi = AAConv2d(self.input_channels, self.hidden_channels, self.kernel_size, dk, dv, Nh)
        self.W_hi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_xf = AAConv2d(self.input_channels, self.hidden_channels, self.kernel_size, dk, dv, Nh)
        self.W_hf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_xc = AAConv2d(self.input_channels, self.hidden_channels, self.kernel_size, dk, dv, Nh)
        self.W_hc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_xo = AAConv2d(self.input_channels, self.hidden_channels, self.kernel_size, dk, dv, Nh)
        self.W_ho = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_ci = nn.Parameter(torch.zeros(1, self.hidden_channels, self.height, self.width, device=torch.device('cuda:0')))
        self.W_cf = nn.Parameter(torch.zeros(1, self.hidden_channels, self.height, self.width, device=torch.device('cuda:0')))
        self.W_co = nn.Parameter(torch.zeros(1, self.hidden_channels, self.height, self.width, device=torch.device('cuda:0')))

    def forward(self, inputs, rep, c):

        i_t = torch.sigmoid(self.W_xi(inputs) + self.W_hi(rep) + c * self.W_ci)
        f_t = torch.sigmoid(self.W_xf(inputs) + self.W_hf(rep) + c * self.W_cf + self.forget_bias)
        c_t = f_t * c + i_t * torch.tanh(self.W_xc(inputs) + self.W_hc(rep))
        o_t = torch.sigmoid(self.W_xo(inputs) + self.W_ho(rep) +  c_t * self.W_co)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def reset_parameters(self):
        self.W_xi.reset_parameters()
        self.W_hi.reset_parameters()
        self.W_xf.reset_parameters()
        self.W_hf.reset_parameters()

        self.W_xc.reset_parameters()
        self.W_hc.reset_parameters()
        self.W_xo.reset_parameters()
        self.W_ho.reset_parameters()
