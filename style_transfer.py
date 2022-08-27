

import torch


from . import models
from . import utils
from torch.optim import LBFGS
from typing import Tuple


def run_style_transfer(
    model: models.StyleTransferNet,
    alpha: float = 1.0,
    beta: float = 1.0e6,
    iters: int = 300,
    lr: float = 1.0,
    display_freq: int = 10,
    dpi: int = 120,
    figsize: Tuple[int] = (4, 4)
) -> torch.Tensor:
    """Runs style transfer."""

    model = model.requires_grad_(False).eval()
    generated_image = torch.randn(
        model.input_size, device=model.device
    ).requires_grad_(True)

    optimizer = LBFGS([generated_image], lr=lr, max_iter=10)
    content_losses, style_losses = [], []
    best_image = generated_image.clone().detach()
    best_loss = float("inf")

    for i in range(iters):
        def closure():
            with torch.no_grad():
                generated_image.clamp_(0, 1)
            optimizer.zero_grad()

            content_layers, style_layers = model(generated_image)
            content_loss = torch.stack(content_layers).sum()
            style_loss = torch.stack(style_layers).sum()

            content_losses.append(content_loss)
            style_losses.append(style_loss)

            loss = alpha * content_loss + beta * style_loss
            loss.backward()
            return loss.item()

        curr_loss = optimizer.step(closure)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_image = generated_image.detach().clone()

        print(
            f"Iter: {i}, Content Loss: {content_losses[-1]:.12f},"
            f"Style Loss {style_losses[-1]:.12f}"
        )
        with torch.no_grad():
            generated_image.clamp_(0, 1)
        if (i + 1) % display_freq == 0:
            utils.display_image(
                generated_image,
                title="Generated Image",
                figsize=figsize,
                dpi=dpi
            )
    return best_image
