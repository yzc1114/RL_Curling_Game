import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.dense1 = nn.Linear(in_features=inputs, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=256)
        self.dense3 = nn.Linear(in_features=256, out_features=outputs)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x