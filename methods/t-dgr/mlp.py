import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input=49, output=4):
    super().__init__()
    self.input = input
    self.layer0 = nn.Linear(input, 512)
    self.layer1 = nn.Linear(512, 512)
    self.layer2 = nn.Linear(512, 512)
    self.layer3 = nn.Linear(512, 512)
    self.layer4 = nn.Linear(512, output)

  def forward(self, x):
    x = nn.functional.relu(self.layer0(x))
    x = nn.functional.relu(self.layer1(x))
    x = nn.functional.relu(self.layer2(x))
    x = nn.functional.relu(self.layer3(x))
    x = self.layer4(x)

    return x
