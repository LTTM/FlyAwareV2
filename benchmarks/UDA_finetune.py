import numpy as np
from os import path
from shutil import rmtree
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50

from utils.args import get_args
from utils.dataset_loader import FLYAWAREDataset, DEFAULT_AUGMENTATIONS, SYNTHETIC_TO_REAL_MAPPING
from utils.losses import MSIW
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

    augs = DEFAULT_AUGMENTATIONS
    DEFAULT_AUGMENTATIONS["resize"] = args.resize

    assert args.pretrained_ckpt is not None, \
            "Cannot run Evaluation without a checkpoint to load."
    assert path.exists(args.pretrained_ckpt), \
            f"Checkpoint [{args.pretrained_ckpt}] does not exist."

    # train set with synthetic data
    tsset = FLYAWAREDataset(root=args.root_path,
                           variant="synthetic",
                           augment_conf=augs,
                           weather=args.weather,
                           town=args.town,
                           height=args.height,
                           modality=args.modality,
                           split='train',
                           minlen=args.iters_per_epoch*args.batch_size)
    tsloader = DataLoader(tsset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          num_workers=args.dloader_workers)

    # train set with real (unlabeled) data
    ttset = FLYAWAREDataset(root=args.root_path,
                            variant="real",
                            augment_conf=augs,
                            weather=args.weather,
                            town=args.town,
                            height=args.height,
                            modality=list(set(args.modality)-{'semantic'}),
                            split='train',
                            minlen=args.iters_per_epoch*args.batch_size,
                            augment=False)
    ttloader = DataLoader(ttset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True,
                          num_workers=args.dloader_workers)

    vset = FLYAWAREDataset(root=args.root_path,
                           variant='real',
                           augment_conf=DEFAULT_AUGMENTATIONS,
                           weather=args.weather,
                           town=args.town,
                           height=args.height,
                           modality=args.modality,
                           split='test',
                           minlen=0)
    vloader = DataLoader(vset,
                         batch_size=1,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         num_workers=args.dloader_workers)

    if args.override_logs:
        rmtree(args.logdir, ignore_errors=True)
    if path.exists(args.logdir):
        raise ValueError("Logging Directory Exists, Stopping."+
                         " If you want to override it turn on the [override_logs] flag.")

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

    # change network to single-channel input
    if 'rgb' not in args.modality:
        with torch.no_grad():
            if args.model == 'mobilenet':
                Co, _, K1, K2 = model.backbone['0'][0].weight.shape
                w = torch.empty(Co, 1, K1, K2)
                torch.nn.init.xavier_uniform_(w)
                model.backbone['0'][0].weight = torch.nn.Parameter(w)
            elif args.model == 'resnet50':
                Co, _, K1, K2 = model.backbone['conv1'].weight.shape
                w = torch.empty(Co, 1, K1, K2)
                torch.nn.init.xavier_uniform_(w)
                model.backbone['conv1'].weight = torch.nn.Parameter(w)

    model.load_state_dict(
        torch.load(args.pretrained_ckpt, "cpu",
                   weights_only=True)
    )
    # update class names and output predictions if coarse-classes are used
    cnames = vset.get_train_label_names()[:-1] if args.class_set == 'fine' \
                else vset.get_coarse_label_names()[:-1]
    if args.class_set == 'coarse':
        out_conv = model.merge if args.model == 'mmlate' else \
                    model.dlv3.classifier[-1] if args.model == 'mmearly' else \
                     model.classifier[-1]
        Co, Ci, K1, K2 = out_conv.weight.shape
        new_weight = torch.empty(len(cnames), Ci, K1, K2)
        new_bias = torch.empty(len(cnames))
        for idx, data in SYNTHETIC_TO_REAL_MAPPING.items():
            if idx > 0:
                new_weight[idx] = out_conv.weight[data["ids"]].mean(dim=0, keepdim=True)
                new_bias[idx] = out_conv.bias[data["ids"]].mean(dim=0, keepdim=True)
        if args.model == 'mmlate':
            model.merge.weight = torch.nn.Parameter(new_weight)
            model.merge.bias = torch.nn.Parameter(new_bias)
        elif args.model == 'mmearly':
            model.dlv3.classifier[-1].weight = torch.nn.Parameter(new_weight)
            model.dlv3.classifier[-1].bias = torch.nn.Parameter(new_bias)
        else:
            model.classifier[-1].weight = torch.nn.Parameter(new_weight)
            model.classifier[-1].bias = torch.nn.Parameter(new_bias)

    model = DataParallel(model)
    model.to(device)

    sup_loss = CrossEntropyLoss(ignore_index=-1)
    sup_loss.to(device)
    uda_loss = MSIW()
    uda_loss.to(device)

    optim = Adam(model.module.parameters(), lr=args.lr)
    gscaler = torch.GradScaler()

    it = 0
    for e in range(args.epochs):
        model.train()
        metrics = Metrics(cnames, device=device)
        for ii, (ssamples, tsamples) in enumerate(tqdm(zip(tsloader, ttloader), total=args.iters_per_epoch,
                desc="Training Epoch %d/%d"%(e+1, args.epochs), smoothing=0)):

            optim.zero_grad()
            lr = cosinescheduler(it, args.epochs*args.iters_per_epoch, args.lr, warmup=args.warmup_iters)
            optim.param_groups[0]['lr'] = lr

            if "rgb" in tsset.modality:
                srgb = ssamples["rgb"].to("cuda")
                trgb = tsamples["rgb"].to("cuda")
            if "depth" in tsset.modality:
                sdth = ssamples["depth"].to("cuda")
                tdth = tsamples["depth"].to("cuda")
                if 'rgb' not in tsset.modality:
                    srgb = sdth
                    trgb = tdth
            if "semantic" in tsset.modality:
                mlb = ssamples["semantic"].to("cuda")
                if args.class_set == "coarse":
                    mlb = vset.label_to_coarse(mlb)
            else:
                raise ValueError("Cannot train models without labels!")

            # if model is ["mmearly", "mmlate"] then rgb and depth are sure to be present
            if args.model == "mmearly":
                srgb = torch.cat([srgb, sdth], dim=1) # make 4D input
                trgb = torch.cat([trgb, tdth], dim=1) # make 4D input

            with torch.autocast(device_type="cuda"):
                if args.model == 'mmlate':
                    sout = model(srgb, sdth)['out']
                    tout = model(trgb, tdth)['out']
                else:
                    sout = model(srgb)['out']
                    tout = model(trgb)['out']
                ls = sup_loss(sout, mlb)
                lu = uda_loss(tout)
                l = ls + args.uda_loss_weight * lu

            gscaler.scale(l).backward()
            gscaler.unscale_(optim)
            for par in model.parameters():
                if par.requires_grad:
                    par.grad = clean_grad(par.grad)
            gscaler.step(optim)
            gscaler.update()

            spred = sout.detach().argmax(dim=1)
            tpred = tout.detach().argmax(dim=1)
            metrics.add_sample(spred, mlb)

            writer.add_scalar('train/lr', lr, it)
            writer.add_scalar('train/loss', l.item(), it)
            writer.add_scalar('train/mIoU', metrics.percent_mIoU(), it)

            optim.step()
            it += 1
            if ii >= args.iters_per_epoch:
                break
            if args.debug:
                break

        writer.add_image('train/source/input', vset.to_rgb(srgb[0,:3].cpu()).permute(1,2,0), it, dataformats="HWC")
        writer.add_image('train/source/label', vset.color_label(mlb[0].cpu(), coarse_level=args.class_set=="coarse"), it, dataformats="HWC")
        writer.add_image('train/source/pred', vset.color_label(spred[0].cpu(), coarse_level=args.class_set=="coarse"), it, dataformats="HWC")
        writer.add_image('train/target/input', vset.to_rgb(trgb[0,:3].cpu()).permute(1,2,0), it, dataformats="HWC")
        writer.add_image('train/target/pred', vset.color_label(tpred[0].cpu(), coarse_level=args.class_set=="coarse"), it, dataformats="HWC")
        torch.save(model.module.state_dict(), args.logdir+"/latest.pth")

        model.eval()
        metrics = Metrics(cnames, device=device)
        with torch.inference_mode():
            for samples in tqdm(vloader, total=len(vloader),
                    desc="Test Epoch %d/%d"%(e+1, args.epochs), smoothing=0):

                if "rgb" in vset.modality:
                    rgb = samples["rgb"].to("cuda")
                if "depth" in vset.modality:
                    dth = samples["depth"].to("cuda")
                    if 'rgb' not in vset.modality:
                        rgb = dth
                if "semantic" in vset.modality:
                    mlb = samples["semantic"].to("cuda")
                    if args.class_set == "coarse":
                        mlb = vset.label_to_coarse(mlb)
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
            writer.add_image('test/input', vset.to_rgb(rgb[0,:3].cpu()).permute(1,2,0), it, dataformats='HWC')
            writer.add_image('test/label', vset.color_label(mlb[0].cpu(), coarse_level=args.class_set=="coarse"), it, dataformats='HWC')
            writer.add_image('test/pred', vset.color_label(pred[0].cpu(), coarse_level=args.class_set=="coarse"), it, dataformats='HWC')
            print(metrics)
