
import torch
import torchvision


# from facenet_pytorch import MTCNN
from pathlib import Path
from .models import LossNet, total_variation_loss
from torch.optim import LBFGS
from tqdm import tqdm
from typing import List, Optional, Tuple
from .utils import display_image, load_image, tensor_to_image

# Default content and style loss layers.
GATYS_CONTENT_DEFAULT = ["conv_4_2"]
GATYS_STYLE_DEFAULT = ["conv_1_1", "conv_2_1", "conv_3_1", "conv_4_1"]


def run_gatys_style_transfer(
    content_src: str,
    style_src: str,
    save_path: Optional[str] = None,
    image_size: Tuple[int, int] = (512, 512),
    content_labels: Optional[List[str]] = None,
    style_labels: Optional[List[str]] = None,
    normalize_input: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0e6,
    num_iters: int = 100,
    lr: float = 1.0,
    tv_reg: float = 1.0e-6,
    device: str = "cpu",
    random: bool = False,
    **kwargs
) -> torch.Tensor:
    """Runs style transfer and returns the resulting image.

    Args:
        content_src (str): Path to content image.
        style_src (str): Path to style image.
        save_path (Optional[str]): Path to save generated image to.
            Default: None.
        image_size (Tuple[int, int]): Shape to resize images.
            Default: (512, 512).
        content_labels (Optional[List[str]]): Layers to calculate content
            losses from. If None is specified, it defaults to conv_4_2.
            Default: None.
        style_labels (Optional[List[str]]): Layers to calculate style losses
            from. If None is specified, it defaults to conv_1_1 through
            conv_4_1.
        normalize_input (bool): Normalizes input using statistics for VGG19.
            Default: True.
        alpha (float): Content loss weight. Default: 1.0.
        beta (float): Style loss weight. Default: 1.0e6.
        num_iters (int): Number of image optimizations. Default: 100.
        lr (float): Learning rate for LBFGS optimizer. Default: 1.0.
        tv_reg (float): Total variation regularization weight. Default: 1.0e-6.
        device (str): Device. Default: 'cpu'.
        random (bool): Initializes generated image with noise. Default: True.

    Returns:
        Tensor: Generated image.
    """

    # Load images.
    content_image = load_image(
        filepath=content_src, size=list(image_size), device=device
    )
    style_image = load_image(
        filepath=style_src, size=list(image_size), device=device
    )

    # Load model and create loss network.
    loss_network = LossNet(
        model=torchvision.models.vgg19(pretrained=True).features.to(device),
        content_image=content_image,
        style_image=style_image,
        content_labels=content_labels or GATYS_CONTENT_DEFAULT,
        style_labels=style_labels or GATYS_STYLE_DEFAULT,
        mean=[0.485, 0.456, 0.406] if normalize_input else [0, 0, 0],
        std=[0.229, 0.224, 0.225] if normalize_input else [1.0, 1.0, 1.0]
    )
    loss_network = loss_network.requires_grad_(False).eval()

    # save network for inference
    if kwargs.get("model_dir", None):
        state_dict = loss_network.cpu().state_dict()
        torch.save({
            "model": state_dict,
            "content_image": loss_network.content_image,
            "style_image": loss_network.style_image,
            "content_labels": loss_network.content_labels,
            "style_labels": loss_network.style_labels,
            "mean": loss_network.mean,
            "std": loss_network.std
        },
            f=str(Path(kwargs.get("model_dir"), "model.pth"))
        )

    # Initialize image.
    if random:
        generated_image = torch.randn(
            content_image.size(), device=device
        ).requires_grad_(True)
    else:
        generated_image = content_image.clone().requires_grad_(True)

    # Load optimizer.
    optimizer = LBFGS([generated_image], lr=lr, max_iter=10)

    content_losses, style_losses = [], []
    best_image, best_loss = None, float("inf")
    display_freq = kwargs.get("display_freq", num_iters // 10)
    epochs = num_iters // display_freq

    print("Starting style transfer...")
    for epoch in range(epochs):
        with tqdm(range(display_freq)) as tq:
            tq.set_description(f"iter [{epoch * display_freq}]")
            for i, _ in enumerate(tq):
                # Define closure for the optimizer.
                def closure():
                    # Force image values between this range.
                    with torch.no_grad():
                        generated_image.clamp_(-1.5, 1.5)
                    optimizer.zero_grad()

                    # Retrieve content and style layers.
                    content_layers, style_layers = loss_network(
                        generated_image
                    )

                    # Calculate perceptual loss.
                    c_loss = s_loss = 0
                    for layer_i in range(len(content_layers)):
                        c_loss += alpha * content_layers[layer_i].loss
                    for layer_j in range(len(style_layers)):
                        s_loss += beta * style_layers[layer_j].loss

                    content_losses.append(c_loss)
                    style_losses.append(s_loss)

                    # Calculate total variation loss.
                    tv_loss = tv_reg * total_variation_loss(generated_image)

                    loss = c_loss + s_loss + tv_loss
                    loss.backward()
                    return loss.item()

                # Optimization step.
                curr_loss = optimizer.step(closure)

                # Display losses.
                tq.set_postfix({
                    "Content Loss": f"{content_losses[-1]:.12f}",
                    "Style Loss": f"{style_losses[-1]:.12f}"
                })

                # Save best image.
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_image = generated_image.detach().clone()

            # Display generated image so far.
            if kwargs.get("display", True):
                display_image(
                    img=generated_image, title="Generated Image"
                )
            if kwargs.get("save_all", False):
                tensor_to_image(generated_image).save(
                    f"{save_path}_iter_{epoch * display_freq}"
                )
    print("Style transfer is complete.")
    # Save image.
    if save_path is not None:
        tensor_to_image(best_image).save(fp=save_path)
        print(f"Saved output to {save_path}.")
    return best_image
