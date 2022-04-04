import torch
import torch.nn as nn


class CIRR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(nIn, nOut, kSize, padding=padding, bias=False)
        self.norm = nn.InstanceNorm3d(nOut, momentum=0.8, eps=1e-03)
        self.act = nn.RReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        output = self.act(output)
        return output


class MTDC(nn.Module):
'''
if stride=1, the MTDC does not perform down-sampling operation
    but only feature extraction steps. 
'''

    def __init__(self, nIn, nOut, kSize, d, stride=1):
        super().__init__()
        n = int(nOut / 4)
        self.c1 = CIRR(nIn, n, k, 1)   # k = 3
        self.conv1 = nn.Conv3d(n, n, kSize, stride=stride, padding=(kSize - 1) / 2, bias=False,
                              dilation=d, groups=2) # d = dilated step
        self.conv2 = nn.Conv3d(n, n, kSize, stride=stride, padding=(kSize - 1) / 2, bias=False,
                              dilation=d, groups=2)
        self.conv3 = nn.Conv3d(n, n, kSize, stride=stride, padding=((kSize - 1) / 2) * 2, bias=False,
                              dilation=2 * d, groups=2)
        self.conv4 = nn.Conv3d(n, n, kSize, stride=stride, padding=((kSize - 1) / 2) * 2, bias=False,
                              dilation=2 * d, groups=2)
        self.norm = nn.InstanceNorm3d(nOut)
        self.act = nn.RRelu(inplace=True)

    def forward(self, input):
        output1 = self.c1(input)
        O1 = self.conv1(output1)
        O2 = self.conv2(output1)
        O3 = self.conv3(output1)
        O4 = self.conv4(output1)

        a1 = O1
        a2 = O1 + o2
        a3 = O1 + o2 + o3

        cat = torch.cat([o4, a1, a2, a3])
        norm = self.norm(cat)
        output = self.act(norm)
        return output