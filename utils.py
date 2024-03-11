
import fiftyone
import fiftyone.zoo as zoo
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from PIL import Image
from typing import List, Tuple, Union

# Statistics for input normalization taken from ImageNet (important
# pre-processing step to correctly extract VGG features)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageDataset(Dataset):
    """Dataset class for wrapping fiftyone zoo datasets.

    Note that this class is meant specifically for fiftyone datasets. To use
    another dataset, you will need to implement your own Dataset class.
    """

    current_split: str
    img_paths: List[str]
    dataset_view: fiftyone.DatasetView

    def __init__(
        self,
        fiftyone_dataset: fiftyone.Dataset,
        img_size: int = 256
    ):
        self.fiftyone_dataset = fiftyone_dataset
        self.splits = zoo.datasets.ZooDatasetInfo(
            zoo_dataset=self.fiftyone_dataset
        ).supported_splits

        # choose a default split
        self.choose_split(self.splits[0])
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize(size=[img_size] * 2),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor

    def choose_split(self, split: str) -> None:
        """Sets the split (e.g. train or test) of the zoo dataset."""
        if split in self.splits:
            # update split
            self.current_split = split
            self.dataset_view = self.fiftyone_dataset.match_tags(split)
            self.img_paths = list(self.dataset_view.values("filepath"))
        else:
            raise ValueError(f"'{split}' is not a valid split.")


def load_coco_zoo_dataset(
    root_dir: str,
    splits: Tuple[str] = ("train", "validation"),
    max_samples: int = 10000,
    img_size: int = 256
) -> ImageDataset:
    """Loads MS COCO fiftyone zoo dataset.

    If necessary, downloads and stores dataset folder under `/root_dir`.
    Otherwise, the dataset is loaded from the given root.
    """
    fiftyone_dataset = zoo.load_zoo_dataset(
        name="coco-2017",
        splits=list(splits),
        max_samples=max_samples,
        shuffle=False,
        dataset_dir=root_dir,
        persistent=True
    )
    return ImageDataset(
        fiftyone_dataset=fiftyone_dataset, img_size=img_size
    )


def denormalize_batch(
    img_tensor: torch.Tensor, mean: List[float], std: List[float]
) -> torch.Tensor:
    """Denormalizes a batch of tensors. Assumes 3 channels (RGB)."""
    batch_size = img_tensor.shape[0]
    mean = torch.tensor(
        mean, dtype=torch.float32
    ).reshape((1, 3, 1, 1)).expand(batch_size, -1)
    std = torch.tensor(
        std, dtype=torch.float32
    ).reshape((1, 3, 1, 1)).expand(batch_size, -1)
    return img_tensor * std + mean


def load_image(filepath: str, size: List[int], device: str) -> torch.Tensor:
    """Loads image as a tensor, given a filepath, device and output size."""
    img = Image.open(fp=filepath)
    img_tensor = F.to_tensor(img).to(device).float()
    img_tensor = F.resize(img_tensor, size=size).unsqueeze(0)
    return img_tensor


def tensor_to_image(img_tensor: torch.Tensor) -> Image.Image:
    """Converts an image tensor to a PIL Image."""
    img_tensor = img_tensor.cpu().clone()
    img_tensor = img_tensor.squeeze(0) if img_tensor.dim() == 4 else img_tensor
    with torch.no_grad():
        img_tensor.clamp_(0, 1)
    img = F.to_pil_image(img_tensor)
    return img


def display_image(
    img: Union[Image.Image, torch.Tensor],
    title: str,
    figsize: Tuple[int] = (3, 3),
    dpi=200
) -> None:
    """Displays an image with a title."""
    img = tensor_to_image(img) if isinstance(img, torch.Tensor) else img
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    plt.imshow(img)
    plt.pause(0.001)
