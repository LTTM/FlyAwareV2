import argparse
from utils.dataset_loader import ALLOWED_HEIGHTS, ALLOWED_MODALITIES, ALLOWED_TOWNS, ALLOWED_WEATHERS

def str2bool(s):
    return s.lower() in ["1", "t", "true", "y", "yes"]

def str2list(s):
    if s == "str":
        return s
    return s.split(",")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default='Z:/datasets/FLYAWARE-V2')
    parser.add_argument("--splits_path", type=str, default='Z:/datasets/FLYAWARE-V2')
    parser.add_argument("--height", type=str2list, default='all', choices=ALLOWED_HEIGHTS)
    parser.add_argument("--modality", type=str2list, default=ALLOWED_MODALITIES)
    parser.add_argument("--town", type=str2list, default='all', choices=ALLOWED_TOWNS)
    parser.add_argument("--weather", type=str2list, default='all', choices=ALLOWED_WEATHERS)
    parser.add_argument("--model", type=str, default='mobilenet', choices=['mobilenet', 'resnet50', 'mmearly', 'mmlate'])

    parser.add_argument("--logdir", type=str, default='logs')
    parser.add_argument("--override_logs", action="store_true", default=False)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--evaldir", type=str, default='evals')

    parser.add_argument("--iters_per_epoch", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--warmup_iters", type=int, default=2000)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dloader_workers", type=int, default=4)

    parser.add_argument("--debug", type=str2bool, default=False)

    return parser.parse_args()
