import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim=784, out_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 

def test():
    model = MLP()
    from torchsummary import summary
    summary(model, (1, 28, 28), device='cpu')

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#Params: ', count_parameters(model))

    model = MLP(3*32*32)
    summary(model, (3, 32, 32), device='cpu')


if __name__ == "__main__":
    test()