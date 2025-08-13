import torch
from torch import nn

class MSIW(nn.Module):
    def __init__(self, ratio=.2):
        super(MSIW, self).__init__()
        self.iw = ratio

    def forward(self, nw_out):
        # extract dimensions
        N, C, H, W = nw_out.shape
        Np = N*H*W # total number of pixels Nx1xHxW == hist.sum()
        # compute probabilities and predicted segmentation map
        prob = torch.softmax(nw_out, dim=1)
        pred = torch.argmax(prob.detach(), dim=1, keepdim=True) # <- argmax, shape N x 1 x H x W
        # compute the predicted class frequencies
        hist = torch.histc(pred, bins=C, min=0, max=C-1) # 1-dimensional vector of length C
        # compute class weights array
        den = torch.clamp(torch.pow(hist, self.iw)*(Np**(1-self.iw)), min=1.)[pred] # <- cast to Nx1xHxW
        # compute the loss
        return -torch.sum(torch.pow(prob, 2)/den)/(N*C)
