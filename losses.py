
from typing import Dict, List

import torch
import torch.backends.cudnn
import torch.nn.functional as F


def gram_matrix(
    feature_maps: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Returns normalized Gram matrix of the given feature maps.

    The gram matrix of a feature map F, where f_k: feature vector k
    (of shape c x (h * w)) is defined as G = FF^T, where G_ij = f_i * f_j,
    the inner product between feature vectors i and j.
    """
    n, c, h, w = feature_maps.shape
    feature_maps = feature_maps.view(n, c, h * w)
    feature_maps_t = feature_maps.transpose(1, 2)
    gram = torch.bmm(feature_maps, feature_maps_t)
    gram /= (c * h * w) if normalize else 1.0
    return gram


def _tv_isotropic(x: torch.Tensor) -> torch.Tensor:
    """Isotropic TV loss."""
    # sum right neighbor pixel differences
    loss = torch.sum((x[:, :, :, :-1] - x[:, :, :, 1:]) ** 2)
    # sum lower neighbor pixel differences
    loss += torch.sum((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2)
    return loss / x.numel()


def _tv_anisotropic(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV loss."""
    # sum right neighbor pixel differences
    loss = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

    # sum lower neighbor pixel differences
    loss += torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return loss / x.numel()


def total_variation_loss(
    image: torch.Tensor, isotropic: bool = True
) -> torch.Tensor:
    """Returns normalized total variation (TV) loss of the given image.

    Implements both (1) `isotropic` and (2) `anisotropic` versions of TV loss,
    where (1) ditches the square root and aggregates the squared derivatives in
    horizontal and vertical directions together, and (2) uses absolute value
    to separately aggregate terms.

    The advantage of (1) is that neither direction (horizontal or vertical) is
    biased, and so edges in both directions are prioritized "equally," whereas
    (2) may prioritize one direction (e.g. vertical) if the loss is
    significantly greater than the other direction.

    See https://en.wikipedia.org/wiki/Total_variation_denoising for more
    details.
    """
    return _tv_isotropic(image) if isotropic else _tv_anisotropic(image)


def style_loss(
    generated_features: torch.Tensor,
    target_features: torch.Tensor
) -> torch.Tensor:
    """Returns the style loss given the generated and target features."""
    generated_gram = gram_matrix(generated_features)
    target_gram = gram_matrix(target_features)
    loss = F.mse_loss(generated_gram, target_gram)
    return loss


def perceptual_loss(
    generated_content: Dict[str, torch.Tensor],
    generated_style: Dict[str, torch.Tensor],
    content_targets: Dict[str, torch.Tensor],
    style_targets: Dict[str, torch.Tensor],
    content_weight: float = 1.0,
    style_weight: float = 1e3,
) -> Dict[str, torch.Tensor]:
    """Calculates the perceptual loss, which combines content and style losses.

    Returns the perceptual loss, as well as the unweighted content and style
    losses.
    """
    c_loss = s_loss = 0
    # calculate total content loss
    for label in content_targets.keys():
        gen_feat = generated_content[label]
        target_feat = content_targets[label]
        c_loss += F.mse_loss(gen_feat, target_feat)

    # calculate total style loss
    for label in style_targets.keys():
        gen_feat = generated_style[label]
        target_feat = style_targets[label]
        s_loss += style_loss(gen_feat, target_feat)

    # combine the losses (perceptual loss)
    loss = content_weight * c_loss + style_weight * s_loss
    return {
        "perceptual": loss, "content": c_loss, "style": s_loss
    }
