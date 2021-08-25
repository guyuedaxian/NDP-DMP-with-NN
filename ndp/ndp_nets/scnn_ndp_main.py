

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmp.utils.utils import init
from dmp.utils import pytorch_util as ptu
from dmp.utils.dmp_layer import DMPIntegrator, DMPParameters
from ndp.mnist_cnn.cnn_net.simple_cnn import SimpleCNNNet


class NdpSCNN(nn.Module):
    """
        NdpCNN network
    """
    def __init__(self,
                 init_w=3e-3,
                 layer_sizes=[784, 500, 100],
                 hidden_activation=F.relu,
                 pt=None,
                 output_activation=torch.tanh,
                 hidden_init=ptu.fanin_init,
                 b_init_value=0.1,
                 state_index=np.arange(1),
                 N=5,  # N of basis functions
                 T=10,  # Rollout length
                 l=10,  #
                 *args,
                 **kwargs
                 ):

        super().__init__()

        self.N = N
        self.l = l
        # 30 * 2 + 2 * 2 = 64
        self.output_size = N * len(state_index) + 2 * len(state_index)
        output_size = self.output_size
        self.T = T
        self.output_activation = output_activation
        self.state_index = state_index
        self.output_dim = output_size

        tau = 1
        dt = 1.0 / (T * self.l)

        self.output_activation = torch.tanh

        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None)
        self.func = DMPIntegrator()

        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)

        # middle layers
        layer_sizes = [784, 500, 100, 500, 2 * output_size, output_size]

        self.hidden_activation = hidden_activation
        self.pt = SimpleCNNNet()
        # load the pre-trained net's parameters
        self.pt.load_state_dict(torch.load(pt, map_location=torch.device('cpu')))
        self.convSize = 4 * 4 * 50
        self.imageSize = 28

        self.middle_layers = []  # 4 layers
        for i in range(1, len(layer_sizes) - 1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            hidden_init(layer.weight)
            layer.bias.data.fill_(b_init_value)
            self.middle_layers.append(layer)
            # add custom module
            self.add_module('middle_layer_' + str(i), layer)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.last_fc = init_(nn.Linear(layer_sizes[-1], output_size))

    def forward(self, input, y0, return_preactivations=False):
        x = input  # [100, 28, 28]
        x = x.view(-1, 1, self.imageSize, self.imageSize) # [100, 1, 28, 28]
        # first conv
        x = F.relu(self.pt.conv1(x))   # [100, 20, 24, 24]
        x = F.max_pool2d(x, 2, 2)      # [100, 20 ,12, 12]

        # full connection layer of CNN
        x = x.view(-1, 12 * 12 * 20)     # [100, 2880]

        activation_fn = self.hidden_activation
        #
        x = self.pt.fc1(x)  # [100, 500]
        x = activation_fn(x)  # [100, 500]

        # many liner layer and through the activation function
        for layer in self.middle_layers:
            # [500, 100] [100, 500] [500, 128] [128, 64]
            x = activation_fn(layer(x))

        # last special fc
        output = self.last_fc(x) * 1000 # [100, 64]

        y0 = y0.reshape(-1, 1)[:, 0]   # [200]
        dy0 = torch.zeros_like(y0) + 0.01   # [200], [0.01000, 0.0100, ...]

        # get the [200, 301], [200, 301] [200, 301]
        # the output of image to the output of DMP.
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0)

        # the goal y
        y = y.view(input.shape[0], len(self.state_index), -1)
        return y.transpose(2, 1)
