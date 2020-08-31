from __future__ import print_function
import torch

# x = torch.empty(5, 3)
# print(x[1, :])  # 索引第2行
# print(x)
# y = torch.rand(5, 3)
# print(x)
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
# x = torch.tensor([5.5, 3])  ##直接从数据构造张量
# print(x)
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 16)  # the size -1 is inferred from other dimensions
# print(x, y, z)
# print(x.size(), y.size(), z.size())


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())  # 学习参数
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

