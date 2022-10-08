import torch
import torch.nn as nn
import torch.nn.functional as F

class CBRR(nn.Module):
    def __init__(self, In, Out, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(In, Out, kSize, stride=stride, padding=padding, bias=False, groups=4)
        self.bn = nn.InstanceNorm3d(Out, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class CB(nn.Module):
    def __init__(self, In, Out, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv3d(In, Out, kSize, stride=stride, padding=padding, bias=False, groups=16)
        self.bn = nn.InstanceNorm3d(Out, momentum=0.95, eps=1e-03)
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output

class BR(nn.Module):
    def __init__(self, Out):
        super().__init__()
        self.bn = nn.InstanceNorm3d(Out, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(inplace=True) 
    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output

class Dilate(nn.Module):
    def __init__(self, In, Out, kSize, stride=1, d=1, groups=2):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv3d(In, Out, kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=16)
        self.bn = nn.InstanceNorm3d(Out, momentum=0.95, eps=1e-03)
        self.act = nn.RReLU(inplace=True)
    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
class MTD(nn.Module): 
    def __init__(self, In, Out, stride=1):
        super().__init__()
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.weight4 = nn.Parameter(torch.ones(1))
        self.conv = nn.Conv3d(In, Out, 1, stride=1, padding=0, bias=False, groups=32)
        self.d1 = Dilate(Out, Out, 3, 1, 1, groups=32)
        self.d2 = Dilate(Out, Out, 3, 1, 1, groups=32)
        self.d4 = Dilate(Out, Out, 3, 1, 2, groups=32)
        self.d8 = Dilate(Out, Out, 3, 1, 2, groups=32)
        self.bn = nn.BatchNorm3d(Out)

    def forward(self, input):
        output1 = self.conv(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d4
        add3 = add2 + d8
        combine = self.bn(self.weight1*d1 + self.weight2*add1 + self.weight3*add2 + self.weight4*add3)
        output = F.relu(combine, inplace=True)
        return output

class MTDC(nn.Module): 
    def __init__(self, In, Out, stride=1):
        super().__init__()
        k = 4
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.weight4 = nn.Parameter(torch.ones(1))
        self.conv = BRR(In, In, 1, 1)
        self.d1 = Dilate(In, Out, 3, stride)
        self.d2 = Dilate(In, Out, 3, stride)
        self.d4 = Dilate(In, Out, 3, stride)
        self.d8 = Dilate(In, Out, 3, stride)
        self.bn = nn.InstanceNorm3d(Out)

    def forward(self, input):
        output1 = self.conv(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        add1 = d1 + d2
        add2 = add1 + d4
        add3 = add2 + d8
        combine = self.bn(self.weight1*d1 + self.weight2*add1 + self.weight3*add2 + self.weight4*add3)
        output = F.relu(combine, inplace=True)

        if input.size() == combine.size():
            combine = input + combine
        return output


class SPC(nn.Module):  # with k=4
    def __init__(self, In, Out, stride=1):
        super().__init__()
        c =  8
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))
        self.weight4 = nn.Parameter(torch.ones(1))
        self.conv = CBRR(In, In, 1, 1)
        self.d1 = CB(In, c, 3, 1)
        self.d2 = CB(In, c, 5, 1)
        self.d4 = CB(In, c, 7, 1)
        self.d8 = CB(In, c, 9, 1)
        self.bn = nn.InstanceNorm3d(Out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        d1 = self.d1(out)
        d2 = self.d2(d1)
        d3 = self.d4(d2)
        d4 = self.d8(d3)
        add1 = d1  + d2
        add2 = add1 + d3
        add3 = add2 + d4
        combine = self.bn(torch.cat([self.weight1*add1, self.weight2*add2, self.weight3*add3, self.weight4*d1], 1))
        output = self.act(combine)
        if input.size() == combine.size():
            combine = input + combine

        return output

class MTA(nn.Module):
    def __init__(self, In, Out, downSize):
        super().__init__()
        self.scale = downSize#0.4
        self.features = CBRR(In, Out, 3, 1)
    def forward(self, x):
        assert x.dim() == 5#[b,c,h,w,d]
        inp_size = x.size()
        out_dim1, out_dim2, out_dim3 = int(inp_size[2] * self.scale), int(inp_size[3] * self.scale), int(inp_size[4] * self.scale)
        x_down = F.adaptive_avg_pool3d(x, output_size=(out_dim1, out_dim2, out_dim3))
        return F.upsample(self.features(x_down), size=(inp_size[2], inp_size[3], inp_size[4]), mode='trilinear')