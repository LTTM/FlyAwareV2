import argparse
from utils.dataset_loader import ALLOWED_HEIGHTS, ALLOWED_MODALITIES, ALLOWED_TOWNS, ALLOWED_WEATHERS

def str2set(s):
    if s == "str":
        return s
    return set(s.split(","))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default='Z:/datasets/FLYAWARE-V2')
    parser.add_argument("--height", type=str2set, default='all')
    parser.add_argument("--modality", type=str2set, default="rgb,semantic")
    parser.add_argument("--town", type=str2set, default='all')
    parser.add_argument("--weather", type=str2set, default='all')
    parser.add_argument("--model", type=str, default='mobilenet', choices=['mobilenet', 'resnet50', 'mmearly', 'mmlate'])

    parser.add_argument("--logdir", type=str, default='logs/test')
    parser.add_argument("--override_logs", action="store_true", default=False)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--evaldir", type=str, default='evals')

    parser.add_argument("--iters_per_epoch", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--warmup_iters", type=int, default=2000)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dloader_workers", type=int, default=8)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")

    args = parser.parse_args()

    assert args.height.intersection(ALLOWED_HEIGHTS) == args.height, \
        f"Illegal height(s) encountered: {args.height-ALLOWED_HEIGHTS}, allowed options: {ALLOWED_HEIGHTS}"
    args.height = list(args.height) # cast back to list

    assert args.modality.intersection(ALLOWED_MODALITIES) == args.modality, \
        f"Illegal modality(ies) encountered: {args.modality-ALLOWED_MODALITIES}, allowed options: {ALLOWED_MODALITIES}"
    args.modality = list(args.modality) # cast back to list

    assert args.town.intersection(ALLOWED_TOWNS) == args.town, \
        f"Illegal town(s) encountered: {args.town-ALLOWED_TOWNS}, allowed options: {ALLOWED_TOWNS}"
    args.town = list(args.town) # cast back to list

    assert args.weather.intersection(ALLOWED_WEATHERS) == args.weather, \
        f"Illegal weather(s) encountered: {args.weather-ALLOWED_WEATHERS}, allowed options: {ALLOWED_WEATHERS}"
    args.weather = list(args.weather) # cast back to list

    if args.model in ["mmearly", "mmlate"]:
        assert args.modality == "all" or \
            set(args.modality).intersection({"rgb", "depth"}) == {"rgb", "depth"}, \
                f"MultiModal require both RGB and Depth modalities to be enabled, you have {args.modality}."

    return args
