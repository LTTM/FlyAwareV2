"""
Module implementing all the need elements to load and handle the
FLYAWARE dataset.

Constants:
    - SYNTHETIC_CLASS_MAPPING: dict mapping synthetic class names to their IDs.
    - REAL_CLASS_MAPPING: dict mapping real class names to their IDs.
    - SYNTHETIC_TO_REAL_MAPPING: dict mapping synthetic class IDs to real class IDs.
    - WEATHER_NAMES_MAPPING: dict mapping real weather names to synthetic ones.
    - ALLOWED_VARIANTS: list of allowed dataset variants (synthetic, real).
    - ALLOWED_WEATHERS: list of allowed weather conditions.
    - ALLOWED_TOWNS: list of allowed towns in the dataset.
    - ALLOWED_HEIGHTS: list of allowed heights for the dataset.
    - ALLOWED_MODALITIES: list of allowed modalities (rgb, depth).
    - DEFAULT_AUGMENTATIONS: default augmentations dictionary to apply during data loading.
    - IMAGENET_MEAN: tensor containing the mean values for the ImageNet dataset.
    - IMAGENET_STD: tensor containing the standard deviation values for the ImageNet dataset.

Classes:
    - FLYAWAREDataset: Class to handle the loading and processing of the FLYAWARE dataset.
"""

from pathlib import Path
from PIL import Image
from typing import Dict, List, Union, Any, Tuple

import torch
from torch.utils.data import Dataset

from torchvision.transforms import v2 as T

# COnstants definition
SYNTHETIC_CLASS_MAPPING = {
    1: {"name": "Building", "color": [70, 70, 70], "train_id": 0},
    2: {"name": "Fence", "color": [190, 153, 153], "train_id": 1},
    3: {"name": "Other", "color": [180, 220, 135], "train_id": 2},
    5: {"name": "Pole", "color": [153, 153, 153], "train_id": 3},
    6: {"name": "RoadLine", "color": [255, 255, 255], "train_id": 4},
    7: {"name": "Road", "color": [128, 64, 128], "train_id": 5},
    8: {"name": "Sidewalk", "color": [244, 35, 232], "train_id": 6},
    9: {"name": "Vegetation", "color": [107, 142, 35], "train_id": 7},
    11: {"name": "Wall", "color": [102, 102, 156], "train_id": 8},
    12: {"name": "Traffic Signs", "color": [220, 220, 0], "train_id": 9},
    13: {"name": "Sky", "color": [70, 130, 180], "train_id": 10},
    14: {"name": "Ground", "color": [81, 0, 81], "train_id": 11},
    15: {"name": "Bridge", "color": [150, 100, 100], "train_id": 12},
    16: {"name": "Rail Track", "color": [230, 150, 140], "train_id": 13},
    17: {"name": "Guard Rail", "color": [180, 165, 180], "train_id": 14},
    18: {"name": "Traffic Light", "color": [250, 170, 30], "train_id": 15},
    19: {"name": "Static", "color": [110, 190, 160], "train_id": 16},
    20: {"name": "Dynamic", "color": [111, 74, 0], "train_id": 17},
    21: {"name": "Water", "color": [45, 60, 150], "train_id": 18},
    22: {"name": "Terrain", "color": [152, 251, 152], "train_id": 19},
    40: {"name": "Person", "color": [220, 20, 60], "train_id": 20},
    41: {"name": "Rider", "color": [255, 0, 0], "train_id": 21},
    100: {"name": "Car", "color": [0, 0, 142], "train_id": 22},
    101: {"name": "Truck", "color": [0, 0, 70], "train_id": 23},
    102: {"name": "Bus", "color": [0, 60, 100], "train_id": 24},
    103: {"name": "Train", "color": [0, 80, 100], "train_id": 25},
    104: {"name": "Motorcycle", "color": [0, 0, 230], "train_id": 26},
    105: {"name": "Bicycle", "color": [119, 11, 32], "train_id": 27},
    0: {"name": "Unlabeled", "color": [0, 0, 0], "train_id": -1},
}
REAL_CLASS_MAPPING = {
    0: {"name": "Building", "color": [128, 0, 0], "train_id": 0},
    1: {"name": "Road", "color": [128, 64, 128], "train_id": 5},
    2: {"name": "Static car", "color": [192, 0, 192], "train_id": 22},
    3: {"name": "Tree", "color": [0, 128, 0], "train_id": 7},
    4: {"name": "Low vegetation", "color": [128, 128, 0], "train_id": 7},
    5: {"name": "Moving car", "color": [64, 0, 128], "train_id": 22},
    6: {"name": "Human", "color": [64, 64, 0], "train_id": 20},
    7: {"name": "Unlabeled", "color": [0, 0, 0], "train_id": -1},
}
SYNTHETIC_TO_REAL_MAPPING = {
    0: {"name": "Building", "ids": [0, 1, 8]},
    1: {"name": "Road", "ids": [4, 5, 6, 12, 13]},
    2: {"name": "Car", "ids": [22, 23, 24, 25, 26, 27]},
    3: {"name": "Vegetation", "ids": [7, 11, 19]},
    4: {"name": "Human", "ids": [20, 21]},
    -1: {"name": "Unlabeled", "ids": [-1, 2, 3, 9, 10, 14, 15, 16, 17, 18]},
}
WEATHER_NAMES_MAPPING = {
    "day": "ClearNoon",
    "night": "ClearNight",
    "rain": "HardRainNoon",
    "fog": "MidFoggyNoon",
}
ALLOWED_VARIANTS = {"synthetic", "real"}
ALLOWED_WEATHERS = {"day", "night", "rain", "fog", "all"}
ALLOWED_TOWNS = {
    "Town01_Opt_120",
    "Town02_Opt_120",
    "Town03_Opt_120",
    "Town04_Opt_120",
    "Town05_Opt_120",
    "Town06_Opt_120",
    "Town07_Opt_120",
    "Town10HD_Opt_120",
    "all",
}
ALLOWED_HEIGHTS = {"height20m", "height50m", "height80m", "all"}
ALLOWED_MODALITIES = {"rgb", "depth", "semantic", "all"}
DEFAULT_AUGMENTATIONS = {
    "resize": 1920, # int|[H:int, W:int]
    "crop": False, # bool|[H:int, W:int]
    "hflip": True,
    "vflip": False,
    "brightness_jitter": .5, # maximum relative brightness change
    "contrast_jitter": .5,   # maximum relative contrast change
    "saturation_jitter": .5, # maximum relative saturation change
    "hue_jitter": .5,        # maximum relative hue change
    "gauss_blur": 1.5,       # maximum sigma of gaussian blur, 0: disabled, min_val: 0.2
    "gauss_noise": 1.5       # std scale of gaussian noise
}
IMAGENET_MEAN = torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32)
IMAGENET_STD = torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32)

class FLYAWAREDataset(Dataset):
    """
    Class to handle the loading and processing of the FLYAWARE dataset.
    """

    def __init__(
        self,
        root: Union[Path, str],
        variant: str,
        augment_conf: Dict[str, Any],
        weather: Union[str, List[str]] = "all",
        town: Union[str, List[str]] = "all",
        height: Union[str, List[str]] = "all",
        modality: Union[str, List[str]] = "rgb",
        split: str = "train",
        minlen: int = 0,
        augment: bool = True,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            root (Union[Path, str]): Path to the dataset root directory.
            variant (str): Dataset variant, either 'synthetic' or 'real'.
            augment_conf (Dict[str, Any]): Configuration for data augmentation,
                        mandatory: 'resize': int|[int,int].
            weather (Union[str, List[str]]): Weather conditions to include.
            town (Union[str, List[str]]): Towns to include in the dataset.
            height (Union[str, List[str]]): Heights to include in the dataset.
            modality (Union[str, List[str]]): Modalities to include (e.g., 'rgb', 'depth').
            split (str): Dataset split, either 'train' or 'val'.
            minlen (int): Minimum length of the dataset, which will be expanded
                          by repeating the samples default is 0 = no change.
            augment (bool): Whether to augment the dataset.

        Raises:
            ValueError: If any of the parameters are not allowed.

        Returns:
            None
        """

        # Parse the input parameters to set
        weather = set(weather) if isinstance(weather, list) else {weather}
        town = set(town) if isinstance(town, list) else {town}
        height = set(height) if isinstance(height, list) else {height}
        modality = set(modality) if isinstance(modality, list) else {modality}

        # Check all the input parameters
        if variant not in ALLOWED_VARIANTS:
            raise ValueError(
                f"Variant '{variant}' is not allowed. Choose from {ALLOWED_VARIANTS}."
            )
        if weather.intersection(ALLOWED_WEATHERS) != weather:
            raise ValueError(
                f"Weather '{weather}' is not allowed. Choose from {ALLOWED_WEATHERS}."
            )
        if town.intersection(ALLOWED_TOWNS) != town:
            raise ValueError(
                f"Town '{town}' is not allowed. Choose from {ALLOWED_TOWNS}."
            )
        if height.intersection(ALLOWED_HEIGHTS) != height:
            raise ValueError(
                f"Height '{height}' is not allowed. Choose from {ALLOWED_HEIGHTS}."
            )
        if modality.intersection(ALLOWED_MODALITIES) != modality:
            raise ValueError(
                f"Modality '{modality}' is not allowed. Choose from {ALLOWED_MODALITIES}."
            )
        if split not in {"train", "test"}:
            raise ValueError(
                f"Split '{split}' is not allowed. Choose either 'train' or 'test'."
            )
        if "resize" not in augment_conf:
            raise ValueError(
                "Augmentation configuration must include 'resize'."
            )
        if not isinstance(augment_conf["resize"], int) and \
            not (isinstance(augment_conf["resize"], list) and \
             len(augment_conf["resize"]) == 2):
            raise ValueError(
                "Resize value must be an integer or a list of two integers."
            )

        # Convert root to Path if it is a string
        if isinstance(root, str):
            root = Path(root)

        # Ensure the root path exists
        if not root.exists():
            raise FileNotFoundError(f"The dataset root path '{root}' does not exist.")

        # Ensure the root path is a directory
        if not root.is_dir():
            raise NotADirectoryError(
                f"The dataset root path '{root}' is not a directory."
            )

        # Initialize the dataset parameters
        self.root = root
        self.variant = variant
        self.weather = weather if weather != {"all"} else ALLOWED_WEATHERS - {"all"}
        self.town = town if town != {"all"} else ALLOWED_TOWNS - {"all"}
        self.height = height if height != {"all"} else ALLOWED_HEIGHTS - {"all"}
        self.modality = (
            modality if modality != {"all"} else ALLOWED_MODALITIES - {"all"}
        )
        self.split = split
        self.minlen = minlen
        self.augment = augment and split == "train"
        self.augment_conf = augment_conf
        self.pil_to_tensor = T.PILToTensor()
        if self.augment_conf["gauss_noise"] > 0:
            self.blur = T.GaussianBlur(5, (0.1, self.augment_conf["gauss_noise"]))
        else:
            self.blur = torch.nn.Identity()
        self.jitter = T.ColorJitter(brightness=self.augment_conf["brightness_jitter"],
                                    contrast=self.augment_conf["contrast_jitter"],
                                    saturation=self.augment_conf["saturation_jitter"],
                                    hue=self.augment_conf["hue_jitter"])

        # Initialize the paths
        self._initialize_items()
        if minlen > 0:
            self._expand_items()

    def get_train_label_names(self) -> List[str]:
        """
        Get the names of all labels in the dataset from the provided constants.
        """
        return [el["name"] for el in SYNTHETIC_CLASS_MAPPING.values()]

    def get_coarse_label_names(self) -> List[str]:
        """
        Get the coarse-level class names from the dataset constants.
        """
        return [el["name"] for el in SYNTHETIC_TO_REAL_MAPPING.values()]

    def get_train_colormap(self) -> List[torch.Tensor]:
        """
        Get the colormap of all labels in the dataset from the provided constants.
        """
        return torch.tensor([el["color"] for el in SYNTHETIC_CLASS_MAPPING.values()], dtype=torch.uint8)

    def get_coarse_colormap(self) -> List[torch.Tensor]:
        """
        Compute the coarse-level colormap from the fine-level one.
        """
        fine_cmap = self.get_train_colormap()
        coarse_cmap = []
        for idx, data in SYNTHETIC_TO_REAL_MAPPING.items():
            if idx != -1:
                source_ids = data["ids"]
                source_cols = torch.stack([fine_cmap[i] for i in source_ids]).float()
                coarse_col = torch.round(torch.mean(source_cols**2, dim=0)**.5).int()
                coarse_cmap.append(coarse_col.tolist())
            else:
                # unlabeled must be black
                coarse_cmap.append([0, 0, 0])
        return torch.tensor(coarse_cmap)

    def _expand_items(self) -> None:
        """
        Expand the dataset items if they are shorter than the minimum length.
        """
        curlen = len(self)
        if curlen < self.minlen:
            expand = self.minlen // curlen + 1
            self.items = {k: v*expand for k,v in self.items.items()}

    def _initialize_items(self) -> None:
        """
        Initializes the dataset itemss based on the initialized parameters.

        Raises:
            FileNotFoundError: If the split file does not exist for the real variant.

        Returns:
            None
        """
        root_path = self.root / self.variant

        # Instantiate the paths
        self.items = {mod: [] for mod in self.modality}

        # Load the paths based on the variant
        if self.variant == "real":
            # Define the root path based on the split
            root_path = root_path / self.split

            for weather in sorted(self.weather):
                # Get the weather-specific path
                weather_path = root_path / weather

                # Get the paths for each modality
                for mod in self.items:
                    # Accept both PNG and JPG formats
                    mod_path = weather_path / mod
                    png_paths = sorted(list(mod_path.glob("*.png")))
                    jpg_paths = sorted(list(mod_path.glob("*.jpg")))
                    self.items[mod].extend(png_paths + jpg_paths)
        else:
            # Read the split txt file
            split_file_path = root_path / f"{self.split}.txt"
            if not split_file_path.exists():
                raise FileNotFoundError(
                    f"The split file '{split_file_path}' does not exist."
                )

            with open(split_file_path, "r", encoding="utf-8") as file:
                frames = [l.strip().zfill(5) for l in file]

            # Load the paths for each modality
            for town in sorted(self.town):
                for weather in sorted(self.weather):
                    weather = WEATHER_NAMES_MAPPING[weather]
                    for height in sorted(self.height):
                        for mod in self.items:
                            base_dir = root_path / town / weather / height / mod
                            for frame in frames:
                                ext = ".png" if mod != "rgb" else ".jpg"
                                self.items[mod].append(base_dir / f"{frame}{ext}")

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.items[list(self.modality)[0]])

    def _resize_tensor(self, tensor: torch.Tensor, interpolation_mode: str) -> torch.Tensor:
        """
        Resize a tensor to the specified size using the given interpolation mode.

        Args:
           tensor (torch.Tensor): The input tensor to resize.
           interpolation_mode (str): The interpolation mode to use.

        Returns:
           torch.Tensor: The resized tensor.
        """
        tensor = tensor.unsqueeze(0)
        if isinstance(self.augment_conf["resize"], int):
            H, W = tensor.shape[2:]
            if H > W:
                W1 = round(self.augment_conf["resize"] * W / H)
                tensor = torch.nn.functional.interpolate(tensor,
                                size=(self.augment_conf["resize"], W1),
                                mode=interpolation_mode)
            else:
                H1 = round(self.augment_conf["resize"] * H / W)
                tensor = torch.nn.functional.interpolate(tensor,
                                size=(H1, self.augment_conf["resize"]),
                                mode=interpolation_mode)
        else:
            tensor = torch.nn.functional.interpolate(tensor, size=self.augment_conf,
                                                    mode=interpolation_mode)
        return tensor.squeeze(0)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            item (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: Sample data.
        """

        # get the modalities that actually have elements
        # (i.e., check if the segmentation labels are presents)
        mods = [k for k, v in self.items.items() if len(v) > 0]

        # load the samples using pillow
        # rgb: RGB image (uint8)
        # depth: Depth image (uint16)
        # semantic: Semantic indices (uint8)
        sample = {k: Image.open(self.items[k][item]) for k in mods}
        sample = self._resize_and_crop(sample)
        if self.augment:
            sample = self._augment_sample(sample)
        if "semantic" in sample:
            sample["semantic"] = sample["semantic"].squeeze(0) # remove channel dimension
        return sample

    def _resize_and_crop(self, sample: Dict[str, Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Resize, crop and convert the sample data to a tensor.

        Args:
           sample (Dict[str, Image.Image]): Sample data.

        Returns:
           Dict[str, torch.Tensor]: Resized and cropped sample data.
        """
        tensors = {k: self.pil_to_tensor(v) for k, v in sample.items()}
        for k in tensors:
            if k == 'rgb':
                # normalize the rgb image with Imagenet mean and std
                tensors[k] = tensors[k].to(torch.float32) / 255.
                tensors[k] -= IMAGENET_MEAN
                tensors[k] /= IMAGENET_STD
                tensors[k] = self._resize_tensor(tensors[k], "bilinear")
            elif k == 'depth':
                # convert to float
                tensors[k] = tensors[k].to(torch.float32) / (2**16 - 1)
                # clamp depth to 2nd max value
                mask = tensors[k] == tensors[k].max()
                tensors[k][mask] = tensors[k][~mask].max()
                # normalize depth to [0, 1]
                tensors[k] -= tensors[k].min()
                tensors[k] /= tensors[k].max()
                # shift it to the same scale as rgb image
                tensors[k] -= IMAGENET_MEAN.mean(dim=0, keepdim=True)
                tensors[k] /= IMAGENET_STD.mean(dim=0, keepdim=True)
                tensors[k] = self._resize_tensor(tensors[k], "bilinear")
            elif k == 'semantic':
                # map the label to training indices
                lb = -1*torch.ones_like(tensors[k], dtype=torch.float32)
                data = SYNTHETIC_CLASS_MAPPING if self.variant == "synthetic" else REAL_CLASS_MAPPING
                idmap = {k: v["train_id"] for k, v in data.items()}
                for rid, tid in idmap.items():
                    lb[tensors[k] == rid] = tid
                tensors[k] = lb
                tensors[k] = self._resize_tensor(tensors[k], "nearest").long()

            if isinstance(self.augment_conf["crop"], list) and len(self.augment_conf["crop"]) == 2:
                H, W = tensors[k].shape[1:] # original height and width
                pH, pW = self.augment_conf["crop"] # patch height and width
                assert H >= pH and W >= pW, "Patch size must be smaller than or equal to original image size."
                cpH, cpW = pH / 2, pW / 2
                if self.augment:
                    cH = torch.randint(round(cpH), round(H - cpH + 1), size=(1,), dtype=torch.long).item()
                    cW = torch.randint(round(cpW), round(W - cpW + 1), size=(1,), dtype=torch.long).item()
                else:
                    cH, cW = H / 2, W / 2
                tensors[k] = tensors[k][:,round(cH - cpH):round(cH + cpH), round(cW - cpW):round(cW + cpW)]
        return tensors

    def _augment_sample(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Augment the sample data.

        Args:
            sample (Dict[str, Image.Image]): Sample data.

        Returns:
            Dict[str, torch.Tensor]: Augmented sample data.
        """
        if self.augment_conf["hflip"] and torch.rand((1,)) < .5:
            sample = {k: v.flip(2) for k, v in sample.items()}
        if self.augment_conf["vflip"] and torch.rand((1,)) < .5:
            sample = {k: v.flip(1) for k, v in sample.items()}

        if "rgb" in sample:
            rgb = sample["rgb"]
            rgb += self.augment_conf["gauss_noise"]*torch.randn_like(rgb)
            rgb = self.blur(rgb)
            rgb = self.jitter(rgb)

        return sample

    @torch.no_grad()
    def color_label(self, label: torch.Tensor, coarse_level: bool = False) -> torch.Tensor:
        """
        Colorize the label tensor.

        Args:
           label (torch.Tensor): Label tensor.
           coarse_level (bool): Wether to use coarse-level classes.

        Returns:
           torch.Tensor: Colorized label tensor.
        """
        if coarse_level:
            cmap = self.get_coarse_colormap()
        else:
            cmap = self.get_train_colormap()
        return cmap[label]

    @torch.no_grad()
    def to_rgb(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Convert the tensor to RGB format, inverting normalization.

        Args:
           ts (torch.Tensor): Tensor to convert.

        Returns:
           torch.Tensor: RGB tensor.
        """
        ts = ts * IMAGENET_STD
        ts = ts + IMAGENET_MEAN
        return torch.clamp(ts, 0, 1)

    @torch.no_grad()
    def label_to_coarse(self, label: torch.Tensor) -> torch.Tensor:
        """
        Convert the label (BxHxW) tensor to coarse-level classes.

        Args:
           label (torch.Tensor): Labels tensor (BxHxW).

        Returns:
           torch.Tensor: Coarse labels tensor.
        """
        coarse_label = -1 * torch.ones_like(label)
        for cidx, data in SYNTHETIC_TO_REAL_MAPPING.items():
            for fidx in data['ids']:
                coarse_label[label==fidx] = cidx
        return coarse_label

    @torch.no_grad()
    def pred_to_coarse(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Convert the predictions (BxCxHxW) tensor to coarse-level classes.

        Args:
           pred (torch.Tensor): Predictions tensor (BxCxHxW).

        Returns:
           torch.Tensor: Coarse predictions tensor.
        """
        B, _, H, W = pred.shape
        C = len(SYNTHETIC_TO_REAL_MAPPING) - 1 # unlabeled is not in the predictions
        coarse_pred = torch.zeros(B,C,H,W, dtype=pred.dtype, device=pred.device)
        for cidx, data in SYNTHETIC_TO_REAL_MAPPING.items():
            if cidx != -1:
                for fidx in data['ids']:
                    coarse_pred[:,cidx] += pred[:,fidx]
        return coarse_pred

    @torch.no_grad()
    def map_to_coarse(self, pred: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the predictions (BxCxHxW) and label (BxHxW) tensors to coarse-level classes.

        Args:
           pred (torch.Tensor): Predictions tensor (BxCxHxW).
           label (torch.Tensor): Labels tensor (BxHxW).

        Returns:
           torch.Tensor: Coarse predictions tensor.
           torch.Tensor: Coarse labels tensor.
        """
        return self.pred_to_coarse(pred), self.label_to_coarse(label)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    aug = DEFAULT_AUGMENTATIONS
    # aug["crop"] = [512, 512]

    # Example usage of the FLYAWAREDataset class
    dataset = FLYAWAREDataset(
        root="Z:/datasets/FLYAWARE-V2",
        variant="real",
        augment_conf=aug,
        weather="all",
        town="all",
        height="all",
        modality="all",
        split='test',
        minlen=0
    )

    sample = dataset[0]

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(dataset.to_rgb(sample['rgb']).permute(1,2,0))
    axs[0].set_title('RGB Image')
    axs[1].imshow(sample['depth'][0])
    axs[1].set_title('Depth Image')
    coarse_label = dataset.label_to_coarse(sample['semantic'])
    axs[2].imshow(
        dataset.color_label(
            coarse_label,
            coarse_level=True
            )
        )
    axs[2].set_title('Labels')
    plt.show()
