
import torch
import torch.nn as nn

class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def test_se():
   x=torch.randn(2,1024,8,8)
   print(x)
   se_block = SEModel(channel=1024)
   output = se_block(x)
   print("Test passed.Output shape:",output.shape)
if __name__ == '__main__':
    test_se()

