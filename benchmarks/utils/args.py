import argparse
from utils.dataset_loader import ALLOWED_HEIGHTS, \
    ALLOWED_MODALITIES, ALLOWED_TOWNS, ALLOWED_WEATHERS, DEFAULT_AUGMENTATIONS

def str2set(s):
    if s == "str":
        return s
    return set(s.split(","))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default='Z:/datasets/FLYAWARE-V2')
    parser.add_argument("--variant", type=str,
                        default="synthetic", choices=["synthetic", "real"],
                        help="Which variant of the FlyAware dataset to load [synthetic/real]. " +
                             "This flag is ignored by the synthetic_pretrain.py script.")
    parser.add_argument("--height", type=str2set, default='all',
                        help="Which height subset to load, valid only for the Synthetic variant.")
    parser.add_argument("--modality", type=str2set, default="rgb,semantic",
                        help="Which modalities to load [rgb/depth/semantic].")
    parser.add_argument("--town", type=str2set, default='all',
                        help="Which town subset to load, valid only for the Synthetic variant.")
    parser.add_argument("--weather", type=str2set, default='all',
                        help="Which weather to load [day/night/rain/fog].")
    parser.add_argument("--model", type=str, default='mobilenet',
                        choices=['mobilenet', 'resnet50', 'mmearly', 'mmlate'])

    parser.add_argument("--logdir", type=str, default='logs/test')
    parser.add_argument("--override_logs", action="store_true")

    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--evaldir", type=str, default='evals')
    parser.add_argument("--eval_tensorboard", action="store_true",
                        help="Wether to also log evaluation metrics in tensorboard.")
    parser.add_argument("--class_set", type=str, default="fine", choices=["coarse", "fine"],
                        help="wether to use the full class set (28 classes), or "+
                             "the coarse-level one (5 classes).")
    parser.add_argument("--finetuned", action="store_true",
                        help="Wether the checkpoint belongs to a finetuned network (5 classes in output).")

    parser.add_argument("--iters_per_epoch", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--warmup_iters", type=int, default=2000)
    parser.add_argument("--uda_loss_weight", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dloader_workers", type=int, default=8)
    parser.add_argument("--resize", type=int, default=DEFAULT_AUGMENTATIONS["resize"])

    parser.add_argument("--debug", action="store_true")
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
        assert args.modality == ["all"] or \
            set(args.modality).intersection({"rgb", "depth"}) == {"rgb", "depth"}, \
                f"MultiModal require both RGB and Depth modalities to be enabled, you have {args.modality}."

    return args
