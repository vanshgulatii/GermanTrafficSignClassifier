import torch
import torch.nn as nn

class TrafficSignCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 43)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

    self.pool = nn.MaxPool2d(2)

    self.fc1 = nn.Linear(64 * 6 * 6, 128)
    self.fc2 = nn.Linear(128, 43)

def forward(self, x):

    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))

    x = x.view(x.size(0), -1)

    x = torch.relu(self.fc1(x))
    x = self.fc2(x)

    return x

