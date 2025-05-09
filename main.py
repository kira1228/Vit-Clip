# Initial python script
import torch.nn as nn
import torch.nn.functional

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.conv3 = nn.Conv2d(20, 30, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2d(x)
        x = F.relu(x)
        return x
