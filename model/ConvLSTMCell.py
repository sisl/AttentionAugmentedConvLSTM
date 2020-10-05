""" Follows the definition from
Convolutional LSTM Network: A Machine Learning Approach for Precipitation
Nowcasting """

"Code based on https://github.com/automan000/Convolutional_LSTM_PyTorch/blob/master/convolution_lstm.py"

import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):

    """
    LSTM:
        x - input
        h - hidden representation
        c - memory cell
        f - forget gate
        o - output gate
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):

        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4 #???

        self.padding = int((kernel_size - 1) / 2) # Padding Check that!

        self.W_i = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_f = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_o = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.W_c = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)

        self.reset_parameters();


    def forward(self, inputs, c):

        i_t = torch.sigmoid(self.W_i(inputs)) #Bias included in self.W_.. initialization
        f_t = torch.sigmoid(self.W_f(inputs))
        o_t = torch.sigmoid(self.W_o(inputs))

        c_t = f_t * c + i_t * torch.tanh(self.W_c(inputs))
        h_t = o_t * torch.tanh(c_t)

        #Add output
        return h_t, c_t


    # Does it need to be like this?
    # Peephole connection issue
    # initialization



    def reset_parameters(self):
        self.W_i.reset_parameters()
        self.W_f.reset_parameters()
        self.W_o.reset_parameters()
        self.W_c.reset_parameters()
