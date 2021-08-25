import torch.nn as nn
import torch.nn.functional as F


class DNNNet(nn.Module):
    """
    A deep neural network with two full connected network.
    """
    def __init__(self):
        super(DNNNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """
        forward function
        """
        x = x.view(-1, 28 * 28 * 1)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
