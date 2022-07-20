import torch.nn.functional as F
import torch.nn as nn

# Simple model based on research by Nvidia
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        y = self.bn(x)
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = F.relu(self.conv5(y))
        y = y.contiguous()
        y = y.view(-1, self.num_flat_features(y))
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = self.output(y)
        return y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        