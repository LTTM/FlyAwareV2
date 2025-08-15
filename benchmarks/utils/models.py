from copy import deepcopy

import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

def get_named_child(module, name):
    stack = [int(tk) if tk.isnumeric() else tk for tk in name.split('.')]
    for depth in stack:
        if isinstance(depth, str):
            module = getattr(module, depth)
        else:
            try:
                module = module[depth]
            except KeyError:
                module = module[str(depth)]
    return module

def set_named_child(module, name, newmodule):
    stack = [int(tk) if tk.isnumeric() else tk for tk in name.split('.')]
    for i, depth in enumerate(stack):
        if i < len(stack)-1:
            if isinstance(depth, str):
                module = getattr(module, depth)
            else:
                try:
                    module = module[depth]
                except KeyError:
                    module = module[str(depth)]
        else:
            if isinstance(depth, str):
                setattr(module, depth, newmodule)
            else:
                try:
                    _ = module[depth]
                    module[depth] = newmodule
                except KeyError:
                    module[str(depth)] = newmodule

class MultiBN(nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.alternate = False

        self.bn1 = deepcopy(bn)
        self.bn2 = deepcopy(bn)

    def forward(self, x):
        if self.alternate:
            return self.bn2(x)
        return self.bn1(x)

class MultiBNModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.alternate = False
        self.bns = [k for k,v in self.model.named_modules() if isinstance(v, nn.BatchNorm2d)]
        # override batchnorms
        for bn in self.bns:
            set_named_child(
                self.model,
                bn,
                MultiBN(
                    get_named_child(
                        self.model,
                        bn
                    )
                )
            )

    def update_alternate(self, alternate=False):
        self.alternate = alternate
        for bn in self.bns:
            m = get_named_child(self.model, bn)
            m.alternate = alternate

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class LateFuse(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.rgb = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        self.dth = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        with torch.no_grad():
            Co, _, K1, K2 = self.dth.backbone['0'][0].weight.shape
            w = torch.empty(Co, 1, K1, K2)
            torch.nn.init.xavier_uniform_(w)
            self.dth.backbone['0'][0].weight = torch.nn.Parameter(w)
        self.merge = nn.Conv2d(2*num_classes, num_classes, 1, bias=False)

    def forward(self, c, d):
        c = self.rgb(c)['out']
        d = self.dth(d)['out']
        x = torch.cat([c,d], dim=1)
        x = self.merge(x)
        return {'out': x}

class EarlyFuse(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dlv3 = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        with torch.no_grad():
            Co, _, K1, K2 = self.dlv3.backbone['0'][0].weight.shape
            w = torch.empty(Co, 4, K1, K2)
            nn.init.xavier_uniform_(w)
            self.dlv3.backbone['0'][0].weight = nn.Parameter(w)

    def forward(self, x):
        return self.dlv3(x)

if __name__ == "__main__":
    EarlyFuse(28)
