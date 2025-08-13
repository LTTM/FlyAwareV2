from os import path, makedirs, remove
from shutil import rmtree
from tqdm import tqdm

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50

from utils.args import get_args
from utils.dataset_loader import FLYAWAREDataset, DEFAULT_AUGMENTATIONS
from utils.metrics import Metrics
from utils.mm_model import EarlyFuse, LateFuse

@torch.no_grad()
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

    vset = FLYAWAREDataset(root=args.root_path,
                           variant=args.variant,
                           augment_conf=DEFAULT_AUGMENTATIONS,
                           weather=args.weather,
                           town=args.town,
                           height=args.height,
                           modality=args.modality,
                           split='test',
                           minlen=0)
    # instantiate dataloader forcing the batch size to 1 if the variant is real
    # necessary since some samples have different aspect ratios
    vloader = DataLoader(vset,
                         batch_size=args.batch_size if args.variant == 'synthetic' else 1,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False,
                         num_workers=args.dloader_workers)

    if args.override_logs:
        if args.eval_tensorboard:
            rmtree(args.evaldir, ignore_errors=True)
        else:
            if path.exists(path.join(args.evaldir, "metrics.csv")):
                remove(path.join(args.evaldir, "metrics.csv"))

    if path.exists(args.evaldir) and args.eval_tensorboard:
        raise ValueError("Evaluation Directory Exists, Stopping."+
                         " If you want to override it turn on the [override_logs] flag.")
    if path.exists(path.join(args.evaldir, "metrics.csv")):
        raise ValueError("Evaluation File Exists, Stopping."+
                         " If you want to override it turn on the [override_logs] flag.")

    if args.eval_tensorboard:
        writer = SummaryWriter(args.evaldir, flush_secs=.5)
    else:
        makedirs(args.evaldir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    nc = 5 if args.finetuned else 28
    if args.model == 'mobilenet':
        # uses stride 16 (https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html)
        model = deeplabv3_mobilenet_v3_large(num_classes=nc)
    elif args.model == 'resnet50':
        model = deeplabv3_resnet50(num_classes=nc)
    elif args.model == 'mmearly':
        model = EarlyFuse(num_classes=nc)
    else:
        model = LateFuse(num_classes=nc)

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
    model = DataParallel(model)
    model.to(device)
    model.eval()

    cnames = vset.get_train_label_names()[:-1] if args.class_set == 'fine' \
                else vset.get_coarse_label_names()[:-1]

    metrics = Metrics(cnames, device=device)
    with torch.inference_mode():
        for samples in tqdm(vloader, total=len(vloader),
                desc="Testing...", smoothing=0):

            if "rgb" in vset.modality:
                rgb = samples["rgb"].to("cuda")
            if "depth" in vset.modality:
                dth = samples["depth"].to("cuda")
                if 'rgb' not in vset.modality:
                    rgb = dth
            if "semantic" in vset.modality:
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

            if args.class_set == "coarse":
                out, mlb = vset.map_to_coarse(out, mlb)

            pred = out.argmax(dim=1)
            metrics.add_sample(pred, mlb)
            if args.debug:
                break

        if args.eval_tensorboard:
            writer.add_scalar('test/mIoU', metrics.percent_mIoU(), 0)
            writer.add_image('test/input', vset.to_rgb(rgb[0,:3].cpu()).permute(1,2,0), 0, dataformats='HWC')
            writer.add_image('test/label', vset.color_label(mlb[0].cpu(), coarse_level=True), 0, dataformats='HWC')
            writer.add_image('test/pred', vset.color_label(pred[0].cpu(), coarse_level=True), 0, dataformats='HWC')

        with open(path.join(args.evaldir, "metrics.csv"), "w", encoding="utf-8") as fout:
            fout.write(metrics.to_csv())

        print(metrics)
