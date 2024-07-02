import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3, 3), stride=2, bias=True)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, bias=False)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(2, 2), bias=False)

        self.conv1d = nn.Conv1d(512, 256, kernel_size=1)
        self.conv1d2 = nn.Conv1d(256, 128, kernel_size=3)
        self.conv1d3 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv1d4 = nn.Conv1d(128, 11, kernel_size=2)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(512)

        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.conv3(x)))

        size = x.size()
        x = x.view(size[0], size[1], -1)
        x = F.relu(self.conv1d(x))
        x = F.relu(self.conv1d2(x))
        x = self.dp2(x)
        x = F.relu(self.conv1d3(x))
        x = F.relu(self.conv1d4(x))

        return x.squeeze()

    @torch.no_grad()
    def predict(self, x):
        pass
