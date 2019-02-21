import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):
    """
    Feed-Forward Neural Network (FFNN) for MNIST.
    Total 4 hidden layers are used as 28*28 -> (1200, 600, 300, 150) -> 10.
    We apply batchnorm and ReLU.
    We add isotropic noise to every hidden layer to stablize training.
    """

    def __init__(self, params):
        super(FFNN, self).__init__()
        self.params = params
        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 600)
        self.fc3 = nn.Linear(600, 300)
        self.fc4 = nn.Linear(300, 150)
        self.fc5 = nn.Linear(150, 10)
        self.bn1 = nn.BatchNorm1d(1200)
        self.bn2 = nn.BatchNorm1d(600)
        self.bn3 = nn.BatchNorm1d(300)
        self.bn4 = nn.BatchNorm1d(150)

    def forward(self, X):
        out = X.view(X.size(0), -1)
        out = F.relu(self.bn1(self.fc1(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn2(self.fc2(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn3(self.fc3(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = F.relu(self.bn4(self.fc4(out)))
        if self.training: out = out + out.clone().normal_(0, 0.5)
        out = self.fc5(out)

        return out