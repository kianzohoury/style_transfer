

import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


from PIL import Image
from typing import List, Tuple, Union


def load_image(filepath: str, size: List[int], device: str) -> torch.Tensor:
    """Loads image as a tensor, given a filepath, device and output size."""
    img = Image.open(fp=filepath)
    img_tensor = TF.to_tensor(img).to(device).float()
    img_tensor = TF.resize(img_tensor, size=size).unsqueeze(0)
    return img_tensor


def tensor_to_image(img_tensor: torch.Tensor) -> Image.Image:
    """Converts an image tensor to a PIL Image."""
    img_tensor = img_tensor.cpu().clone()
    img_tensor = img_tensor.squeeze(0) if img_tensor.dim() == 4 else img_tensor
    img = TF.to_pil_image(img_tensor)
    return img


def display_image(
    img: Union[Image.Image, torch.Tensor],
    title: str,
    figsize: Tuple[int] = (6, 6),
    dpi=200
):
    """Displays an image with a title."""
    img = tensor_to_image(img) if isinstance(img, torch.Tensor) else img
    plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    plt.imshow(img)
    plt.pause(0.001)
