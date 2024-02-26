
import shutil
import torch
import torchvision


from pathlib import Path
from PIL import Image
from .models.models import VGGLossNet, perceptual_loss
from torch.optim import LBFGS
from torchvision.models import VGG16_Weights, VGG16_BN_Weights
from typing import List, Optional, Tuple, Union
from .utils import display_image, load_image, tensor_to_image

# Default content and style loss layers.
_VGG_CONTENT_DEFAULT = ["conv_4_2"]
_VGG_STYLE_DEFAULT = ["conv_1_1", "conv_2_1", "conv_3_1", "conv_4_1"]
_VGG_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_VGG_IMAGENET_STD = [0.229, 0.224, 0.225]


def run_gatys_style_transfer(
    content_src: str,
    style_src: str,
    image_size: Union[int, Tuple[int, int]] = (512, 512),
    content_labels: Optional[List[str]] = None,
    style_labels: Optional[List[str]] = None,
    normalize_input: bool = True,
    alpha: float = 1.0,
    beta: float = 1e3,
    lbfgs_iters: int = 10,
    num_steps: int = 50,
    lr: float = 1e-3,
    tv_reg: float = 1e-6,
    device: str = "cpu",
    init_noise: bool = False,
    save_fp: Optional[str] = None,
    save_gif: bool = False,
    save_all: bool = False,
    display: bool = False
) -> Image:
    """Runs Gatys et al. neural style transfer method and returns the resulting image.

    Args:
        content_src (str): Path to content image.
        style_src (str): Path to style image.
        image_size (Tuple[int, int]): Shape to resize images.
            Default: (512, 512).
        content_labels (Optional[List[str]]): Layers to calculate content
            losses from. If None is specified, it defaults to conv_4_2 in VGG16.
            Default: None.
        style_labels (Optional[List[str]]): Layers to calculate style losses
            from. If None is specified, it defaults to conv_1_1 through
            conv_4_1 in VGG16.
        normalize_input (bool): Normalizes input using statistics for VGG19.
            Default: True.
        alpha (float): Content loss weight. Default: 1.0.
        beta (float): Style loss weight. Default: 1e3.
        lbfgs_iters (int): Max number of L-BFGS iterations per optimization step. Default: 10.
        num_steps (int): Number of image optimizations, resulting in a maximum of
            `lbfgs_iters` * `num_steps` total L-BFGS iterations. Default: 50.
        lr (float): Learning rate for L-BFGS optimizer. Default: 1e-3.
        tv_reg (float): Total variation regularization weight. Default: 1e-6.
        device (str): Device. Default: 'cpu'.
        init_noise (bool): Initializes generated image with noise. Default: False.
        save_fp (Optional[str]): Path to save generated image, using a valid format (e.g. jpg, tiff).
            Default: None.
        save_gif (bool): If True, saves a .gif version of the image saved under `save_fp`. Default: False.
        save_all (bool): If True, saves a folder of all intermediate images in the same directory as `save_fp`.
            Default: False.
        display (bool): Whether to display intermediate images after each optimization step. Default: True.

    Returns:
        Image: Final generated image.
    """

    # Load images.
    content_image = load_image(
        filepath=content_src, size=list(image_size), device=device
    )
    style_image = load_image(
        filepath=style_src, size=list(image_size), device=device
    )

    # Load pretrained loss network (VGG16)
    loss_network = VGGLossNet(
        model=torchvision.models.vgg16(weights=VGG16_Weights).features.to(device),
        content_image=content_image,
        style_image=style_image,
        content_labels=content_labels or _VGG_CONTENT_DEFAULT,
        style_labels=style_labels or _VGG_STYLE_DEFAULT,
        mean=[0.485, 0.456, 0.406] if normalize_input else [0, 0, 0],
        std=[0.229, 0.224, 0.225] if normalize_input else [1.0, 1.0, 1.0]
    )
    loss_network = loss_network.requires_grad_(False).eval()

    # Initialize image.
    if init_noise:
        generated_image = torch.randn(
            content_image.size(), device=device
        ).requires_grad_(True)
    else:
        generated_image = content_image.clone().requires_grad_(True)

    # Load optimizer.
    optimizer = LBFGS([generated_image], lr=lr, max_iter=lbfgs_iters)

    all_images, best_image, best_loss = [], None, float("inf")
    print("Starting style transfer using direct optimization...")
    for step in range(num_steps):
        print(
            f"Step [{step + 1}/{num_steps}], "
            f"Iterations [{step * lbfgs_iters}/{num_steps * lbfgs_iters}]"
        )
        content_loss = style_loss = tv_loss = 0

        # Define closure for the optimizer.
        def closure():
            # allows us to update losses outside of closure
            nonlocal content_loss, style_loss, tv_loss

            # Force image values to be between 0 and 1.
            with torch.no_grad():
                generated_image.clamp_(0, 1)
            optimizer.zero_grad()

            # Run forward pass through loss network.
            _ = loss_network(
                generated_image
            )

            # Calculate perceptual loss.
            loss, content_loss, style_loss, tv_loss = perceptual_loss(
                content_losses=loss_network.content_layers,
                style_losses=loss_network.style_layers,
                content_weight=alpha,
                style_weight=beta,
                generated_img=generated_image,
                tv_weight=tv_reg
            )
            loss.backward()
            return loss.item()

        # Optimization step.
        curr_loss = optimizer.step(closure)

        # Display losses.
        print(
            f"Total Loss: {curr_loss:.6f}\n"
            f"Content Loss: {content_loss:.6f}\n"
            f"Style Loss: {style_loss:.6f}\n"
            f"TV Loss: {style_loss:.6f}"
        )

        # Save best image.
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_image = tensor_to_image(generated_image.detach().clone())

        # Display generated image so far.
        if display:
            display_image(
                img=generated_image, title="Generated Image"
            )
        if save_all or save_gif:
            all_images.append(tensor_to_image(generated_image))
    print("Style transfer is complete.")
    if save_fp:
        best_image.save(save_fp)
        print(f"Saved result to {save_fp}.")
    if save_all:
        save_dir = Path(save_fp).parent / Path(save_fp).stem + "_all"
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True)
        else:
            shutil.rmtree(save_dir)

        # save all intermediate images
        for i in range(len(all_images)):
            all_images[i].save(
                save_dir / Path(save_fp).stem + f"_{(i + 1) * lbfgs_iters}" + Path(save_fp).suffix
            )
        print(f"Saved all outputs to {save_dir}.")
    if save_gif:
        if not init_noise:
            starting_image = tensor_to_image(content_image)
        else:
            starting_image = all_images.pop(0)
        starting_image.save(
            fp=Path(save_fp).parent / Path(save_fp).stem + ".gif",
            save_all=True,
            append_images=all_images,
            duration=100,
            loop=0
        )
    return best_image
