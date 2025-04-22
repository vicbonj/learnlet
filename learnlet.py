import torch
import torch.nn as nn
import torch.nn.functional as F


class Starlet(nn.Module):
    def __init__(self, n_scales=5, normalized_square=False, padding_mode='reflect'):
        super(Starlet, self).__init__()

        dilat = 1
        kernel = torch.FloatTensor([1/16, 1/4, 3/8, 1/4, 1/16])[None,:]
        kernel2d = torch.matmul(kernel.T, kernel)[None,None,:]

        if normalized_square is True:
            kernel2d /= torch.sum(kernel2d**2, axis=(1, 2, 3)).sqrt()

        pad = int((kernel.shape[1]-1)/2)

        self.convs = nn.ModuleList()
        for i in range(n_scales-1):
            self.convs.append(nn.Conv2d(1, 1, kernel.shape[1], bias=False, padding=pad*dilat, dilation=dilat, padding_mode=padding_mode))
            self.convs[i].weight = torch.nn.Parameter(kernel2d, requires_grad=False)
            dilat *= 2

    def forward(self, x):
        wt = []
        for i in range(len(self.convs)):
            x_new = self.convs[i](x)
            wt.append(x-x_new)
            x = x_new
        wt.append(x_new)

        return torch.cat(wt, axis=1)


class Learnlet(nn.Module):
    def __init__(self, n_scales=5, kernel_size_A=11, kernel_size_S=11, filters=64, exact_rec=False, thresh='hard', model='squared'):
        super(Learnlet, self).__init__()

        self.exact_rec = exact_rec
        self.thresh = thresh
        self.model = model
        dilat = 1
        pad_A = int((kernel_size_A-1)/2)
        pad_S = int((kernel_size_S-1)/2)
        self.id_filter = torch.zeros(kernel_size_A, kernel_size_A).cuda()
        self.id_filter[pad_A, pad_A] = 1

        if exact_rec is True:
            filters_s = filters
        else:
            filters_s = filters + 1

        self.convs_A = nn.ModuleList()
        self.convs_S = nn.ModuleList()
        for i in range(n_scales-1):
            self.convs_A.append(nn.Conv2d(1, filters+1, kernel_size_A, bias=False, padding=pad_A*dilat, dilation=dilat, padding_mode='reflect'))
            self.convs_S.append(nn.Conv2d(filters_s, filters_s, kernel_size_S, bias=False, padding=pad_S*dilat, dilation=dilat, padding_mode='reflect', groups=filters_s))
            dilat *= 2

    def forward(self, x, ks):
        wt = []
        for i in range(len(self.convs_A)):
            x_in = x[:,i][:,None]
            if self.model == 'squared':
                self.convs_A[i].weight.data /= torch.sum(self.convs_A[i].weight.data**2, axis=(1, 2, 3), keepdim=True).sqrt()
            elif self.model == 'sum':
                self.convs_A[i].weight.data /= torch.sum(self.convs_A[i].weight.data, axis=(1, 2, 3), keepdim=True)
            else:
                print(error)
            self.convs_A[i].weight.data[0,0] = torch.nn.Parameter(self.id_filter, requires_grad=False)

            x_a = self.convs_A[i](x_in)
            if self.model == 'squared':
                thresh = ks[:,i][:,None,None,None]*torch.var(x_in, axis=(1, 2, 3)).sqrt()[:,None,None,None]
            elif self.model == 'sum':
                thresh = ks[:,i][:,None,None,None]*torch.var(x_a, axis=(2, 3)).sqrt()[:,:,None,None]
            else:
                print(error)
            if self.thresh == 'hard':
                x_a_t = x_a * torch.sigmoid((torch.abs(x_a) - thresh) / 1e-3)
            elif self.thresh == 'soft':
                x_a_t = torch.sign(x_a) * torch.relu(torch.abs(x_a) - thresh)
            else:
                print('Not implemented thresholding')
            if self.exact_rec is True:
                x_in_t = x_a_t[:,0][:,None]

                x_s = self.convs_S[i](x_a_t[:,1:])

                x_a_id = self.convs_A[i](x_in_t)
                x_s_id = self.convs_S[i](x_a_id[:,1:])

                last_one = x_in_t - torch.sum(x_s_id, axis=1)[:,None]

                x_s = torch.cat([last_one, x_s], axis=1)
            else:
                x_s = self.convs_S[i](x_a_t)

            x_s = torch.sum(x_s, axis=1)[:,None]
            wt.append(x_s)

        return torch.cat(wt, axis=1)

class MiniNet_multi(nn.Module):
    def __init__(self, n=4):
        super(MiniNet_multi, self).__init__()

        self.dense = nn.Linear(1, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, n)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.last_act(x)

        return x
