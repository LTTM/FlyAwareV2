import numpy as np
from os import path
from shutil import rmtree
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50

from utils.args import get_args
from utils.dataset_loader import FLYAWAREDataset, DEFAULT_AUGMENTATIONS
from utils.metrics import Metrics
from utils.mm_model import EarlyFuse, LateFuse

def cosinescheduler(it, niters, baselr=2.5e-4, warmup=2000):
    if it <= warmup:
        return baselr*it/warmup
    it -= warmup
    scale = np.cos((it/(niters-warmup))*(np.pi/2))**2
    return scale*baselr

@torch.no_grad()
def clean_grad(grad, m_norm=1.):
    grad[torch.isnan(grad)] = 0
    grad[torch.isinf(grad)] = 0
    norm = grad.norm()
    if norm > m_norm:
        grad /= norm
    return grad

def set_seed(seed):
    # temporarily import random and numpy-random
    # to seed them as well, some other piece of
    # code may be using their RNGs inside
    import random # pylint: disable=import-outside-toplevel
    from numpy import random as npr # pylint: disable=import-outside-toplevel
    random.seed(seed)
    npr.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    tset = FLYAWAREDataset(root=args.root_path,
                           variant="synthetic",
                           augment_conf=DEFAULT_AUGMENTATIONS,
                           weather=args.weather,
                           town=args.town,
                           height=args.height,
                           modality=args.modality,
                           split='train',
                           minlen=args.iters_per_epoch*args.batch_size)
    tloader = DataLoader(tset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=True,
                         num_workers=args.dloader_workers)

    vset = FLYAWAREDataset(root=args.root_path,
                           variant="synthetic",
                           augment_conf=DEFAULT_AUGMENTATIONS,
                           weather=args.weather,
                           town=args.town,
                           height=args.height,
                           modality=args.modality,
                           split='test',
                           minlen=0)
    vloader = DataLoader(vset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         num_workers=args.dloader_workers)

    if args.override_logs:
        rmtree(args.logdir, ignore_errors=True)
    if path.exists(args.logdir):
        raise ValueError("Logging Directory Exists, Stopping. If you want to override it turn on the [override_logs] flag.")

    writer = SummaryWriter(args.logdir, flush_secs=.5)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if args.model == 'mobilenet':
        # uses stride 16 (https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html)
        model = deeplabv3_mobilenet_v3_large(num_classes=28)
    elif args.model == 'resnet50':
        model = deeplabv3_resnet50(num_classes=28)
    elif args.model == 'mmearly':
        model = EarlyFuse(num_classes=28)
    else:
        model = LateFuse(num_classes=28)
    model.to(device)

    loss = CrossEntropyLoss(ignore_index=-1)
    loss.to(device)

    optim = Adam(model.parameters(), lr=args.lr)
    gscaler = torch.GradScaler()

    it = 0
    for e in range(args.epochs):
        model.train()
        metrics = Metrics(tset.get_train_label_names()[:-1], device=device)
        for ii, samples in enumerate(tqdm(tloader, total=args.iters_per_epoch, desc="Training Epoch %d/%d"%(e+1, args.epochs))):
            optim.zero_grad()
            lr = cosinescheduler(it, args.epochs*args.iters_per_epoch, args.lr, warmup=args.warmup_iters)
            optim.param_groups[0]['lr'] = lr

            if "rgb" in tset.modality:
                rgb = samples["rgb"].to("cuda")
            if "depth" in tset.modality:
                dth = samples["depth"].to("cuda")
            if "semantic" in tset.modality:
                mlb = samples["semantic"].to("cuda")
            else:
                raise ValueError("Cannot train models without labels!")

            # if model is ["mmearly", "mmlate"] then rgb and depth are sure to be present
            if args.model == "mmearly":
                dth = dth.mean(dim=1, keepdim=True) # squeeze to single channel
                rgb = torch.cat([rgb, dth], dim=1) # make 4D input

            with torch.autocast(device_type="cuda"):
                if args.model == 'mmlate':
                    out = model(rgb, dth)['out']
                else:
                    out = model(rgb)['out']
                l = loss(out, mlb)
            gscaler.scale(l).backward()
            gscaler.unscale_(optim)
            for par in model.parameters():
                if par.requires_grad:
                    par.grad = clean_grad(par.grad)
            gscaler.step(optim)
            gscaler.update()

            pred = out.detach().argmax(dim=1)
            metrics.add_sample(pred, mlb)

            writer.add_scalar('train/lr', lr, it)
            writer.add_scalar('train/loss', l.item(), it)
            writer.add_scalar('train/mIoU', metrics.percent_mIoU(), it)

            optim.step()
            it += 1
            if ii >= args.iters_per_epoch:
                break
            if args.debug:
                break

        writer.add_image('train/input', tset.to_rgb(rgb[0,:3].cpu()).permute(1,2,0), it, dataformats="HWC")
        writer.add_image('train/label', tset.color_label(mlb[0].cpu()), it, dataformats="HWC")
        writer.add_image('train/pred', tset.color_label(pred[0].cpu()), it, dataformats="HWC")
        torch.save(model.state_dict(), args.logdir+"/latest.pth")

        model.eval()
        metrics = Metrics(tset.get_train_label_names()[:-1], device=device)
        with torch.inference_mode():
            for samples in tqdm(vloader, total=len(vloader), desc="Test Epoch %d/%d"%(e+1, args.epochs)):
                if "rgb" in tset.modality:
                    rgb = samples["rgb"].to("cuda")
                if "depth" in tset.modality:
                    dth = samples["depth"].to("cuda")
                if "semantic" in tset.modality:
                    mlb = samples["semantic"].to("cuda")
                else:
                    raise ValueError("Cannot evaluate models without labels!")

                # if model is ["mmearly", "mmlate"] then rgb and depth are sure to be present
                if args.model == "mmearly":
                    dth = dth.mean(dim=1, keepdim=True) # squeeze to single channel
                    rgb = torch.cat([rgb, dth], dim=1) # make 4D input

                with torch.autocast(device_type="cuda"):
                    if args.model == 'mmlate':
                        out = model(rgb, dth)['out']
                    else:
                        out = model(rgb)['out']

                pred = out.argmax(dim=1)
                metrics.add_sample(pred, mlb)
                if args.debug:
                    break

            writer.add_scalar('test/mIoU', metrics.percent_mIoU(), it)
            writer.add_image('test/input', tset.to_rgb(rgb[0,:3].cpu()).permute(1,2,0), it, dataformats='HWC')
            writer.add_image('test/label', tset.color_label(mlb[0].cpu()), it, dataformats='HWC')
            writer.add_image('test/pred', tset.color_label(pred[0].cpu()), it, dataformats='HWC')
            print(metrics)
