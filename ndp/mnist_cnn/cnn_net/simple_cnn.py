import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNNet(nn.Module):
    """
    A simple convolutional neural network with one layer.
    """
    def __init__(self):
        super(SimpleCNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.fc1 = nn.Linear(12 * 12 * 20, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """
        forward function
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 12 * 12 * 20)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
