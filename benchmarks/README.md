# Benchmarks

This repository contains all the code and tools needed to evaluate the FlyAwareV2 dataset.

## How to run the code

1. Pretrain on FlyAwareV2 synthetic:
    - RGB Town10 only:

        `python synthetic_pretrain.py --root_path "<path_to_dataset>" --logdir "logs/mnet_allh_t10_allw_rgb" --batch_size 16 --dloader_workers 6 --town Town10_Opt_120`

    - RGB 20m height only:

        `python synthetic_pretrain.py --root_path "<path_to_dataset>" --logdir "logs/mnet_h20_allt_allw_rgb" --batch_size 16 --dloader_workers 6 --height height20m`

    - RGB night only:

        `python synthetic_pretrain.py --root_path "<path_to_dataset>" --logdir "logs/mnet_allh_allt_night_rgb" --batch_size 16 --dloader_workers 6 --weather night`

    - Early-Fusion MultiModal:

        `python synthetic_pretrain.py --root_path "<path_to_dataset>" --logdir "logs/mnet_allh_allt_allw_mmearly" --batch_size 16 --dloader_workers 6 --model mmearly --modality all`

2. The model argument can switch between:
    - `mobilenet`: MobileNetV3 + DeepLabV3 with 3-/1-channel input for RGB/Depth modalities
    - `resnet50`: ResNet50 + DeepLabV3 with 3-/1-channel input for RGB/Depth modalities
    - `mmearly`: MultiModal MobileNetV3 + DeepLabV3 with 4-channel input (RGB+Depth)
    - `mmlate`: MultiModal 2x MobileNetV3 + DeepLabV3 two-encoder latent-fusion multimodal-network

3. After training the final checkpoint will be stored on `<logdir>/latest.pth`. You can use it to infer on any subset of FlyAwareV2 using `test.py <args>`.
    - DayTime fine-level classes inference on real data:

        `test.py --root_path "<path_to_dataset>" --variant real --pretrained_ckpt "logs/mnet_allh_allt_day_rgb/latest.pth" --weather day --class_set fine`

4. Fine-tune the checkpoints on real data using UDA techniques.
    `finetune_uda.py --root_path "<path_to_dataset>" --pretrained_ckpt "logs/mnet_allh_allt_allw_rgb/latest.pth" --logdir "logs/rgb_only_finetuned"`

## Credits

This project was created by:

- Francesco Barbato
- Matteo Caligiuri

Dipartimento di Ingegneria dell'Informazione (DEI) - UniPD
