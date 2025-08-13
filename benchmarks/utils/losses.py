import numpy as np
import torch
from torch import nn

class MSIW(nn.Module):
    def __init__(self, ratio=.2):
        super(MSIW, self).__init__()
        self.iw = ratio

    def forward(self, nw_out):
        # extract dimensions
        N, C, H, W = nw_out.shape
        with torch.no_grad():
            Np = N*H*W # total number of pixels Nx1xHxW == hist.sum()
            pred = torch.argmax(nw_out.detach(), dim=1, keepdim=True) # <- argmax, shape N x 1 x H x W
            # compute the predicted class frequencies
            hist = torch.histc(pred.float(), bins=C, min=0, max=C-1) # 1-dimensional vector of length C
            # compute class weights array
            den = torch.clamp(torch.pow(hist, self.iw)*np.power(Np, 1-self.iw), min=1.)[pred] # <- cast to Nx1xHxW
        # compute probabilities and predicted segmentation map
        prob = torch.softmax(nw_out, dim=1)
        # scale probabilities
        prob = torch.pow(prob, 2)/den
        # compute the loss
        return -torch.sum(prob)/(N*C)

if __name__ == "__main__":
    l = MSIW()
    o = torch.ones(16,5,1080,1920, dtype=torch.float16)
    print(l(o))
