

import torch
import torch.nn as nn
import torchvision


from . import models
from .models import total_variation_loss
from . import utils
from torch.optim import LBFGS, AdamW
from typing import List, Tuple
from tqdm import tqdm


def run_style_transfer(
    network: nn.Module,
    input_mean: List[float],
    input_std: List[float],
    content_src: str,
    style_src: str,
    image_size: List[int],
    content_labels: List[str],
    style_labels: List[str],
    alpha: float = 1.0,
    beta: float = 1.0e6,
    num_iters: int = 300,
    lr: float = 1.0,
    tv_reg: float = 1.0e-6,
    device: str = "cpu",
    **kwargs
) -> torch.Tensor:
    """Runs style transfer."""

    # Load images.
    content_image = utils.load_image(
        filepath=content_src, size=image_size, device=device
    )
    style_image = utils.load_image(
        filepath=style_src, size=image_size, device=device
    )

    # Load model.
    model = models.StyleTransferNet(
        model=network.to(device).eval(),
        content_image=content_image,
        style_image=style_image,
        content_labels=content_labels,
        style_labels=style_labels,
        mean=input_mean,
        std=input_std
    )
    model = model.requires_grad_(False).eval()

    # Initialize image, load optimizer.
    generated_image = torch.randn(
        model.input_size, device=device
    ).requires_grad_(True)
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
                    content_layers, style_layers = model(generated_image)

                    # Calculate perceptual loss.
                    c_loss = s_loss = 0
                    for layer_i in range(len(content_layers)):
                        c_loss += alpha * content_layers[layer_i].loss
                    for layer_i in range(len(style_layers)):
                        s_loss += beta * style_layers[layer_i].loss
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
            if not kwargs.get("display", True):
                utils.display_image(
                    img=generated_image, title="Generated Image"
                )
    return best_image
