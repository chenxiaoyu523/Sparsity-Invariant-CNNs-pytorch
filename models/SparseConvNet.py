import torch.nn as nn
import torch


class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()

        padding = kernel_size//2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)


        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding)

        

    def forward(self, x, mask):
        x = x*mask
        x = self.conv(x)
        normalizer = 1/(self.sparsity(mask)+1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.relu(x)
        
        mask = self.max_pool(mask)

        return x, mask



class SparseConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.SparseLayer1 = SparseConv(1, 16, 11)
        self.SparseLayer2 = SparseConv(16, 16, 7)
        self.SparseLayer3 = SparseConv(16, 16, 5)
        self.SparseLayer4 = SparseConv(16, 16, 3)
        self.SparseLayer5 = SparseConv(16, 16, 3)
        self.SparseLayer6 = SparseConv(16, 1, 1)

    def forward(self, x, mask):
        
        x, mask = self.SparseLayer1(x, mask)
        x, mask = self.SparseLayer2(x, mask)
        x, mask = self.SparseLayer3(x, mask)
        x, mask = self.SparseLayer4(x, mask)
        x, mask = self.SparseLayer5(x, mask)
        x, mask = self.SparseLayer6(x, mask)

        return x
