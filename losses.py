
from typing import Dict, List

import torch
import torch.backends.cudnn
import torch.nn.functional as F


def gram_matrix(feature_maps: torch.Tensor) -> torch.Tensor:
    """Returns normalized Gram matrix of the given feature maps."""
    n, c, h, w = feature_maps.shape
    feature_maps = feature_maps.reshape((n, c, h * w))
    feature_maps_t = torch.transpose(feature_maps, 1, 2)
    gram = torch.bmm(feature_maps, feature_maps_t)
    return gram / (c * h * w)


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    """Returns total variation (TV) loss of the given image.

    Implements isotropic (and differentiable) version of TV loss, without
    using square root to aggregate squared differences.
    """
    # sum right neighbor pixel differences
    loss = torch.sum(
        (image[:, :, :, :-1] - image[:, :, :, 1:]) ** 2
    )
    # sum lower neighbor pixel differences
    loss += torch.sum(
        (image[:, :, :-1, :] - image[:, :, 1:, :]) ** 2
    )
    return loss


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
    generated_content: List[torch.Tensor],
    generated_style: List[torch.Tensor],
    content_targets: List[torch.Tensor],
    style_targets: List[torch.Tensor],
    content_weight: float = 1.0,
    style_weight: float = 1e3,
) -> Dict[str, torch.Tensor]:
    """Calculates the perceptual loss, which combines content and style losses.

    Returns the perceptual loss, as well as the unweighted content and style
    losses.
    """
    c_loss = s_loss = 0
    # calculate total content loss
    for gen_feat, target_feat in zip(generated_content, content_targets):
        c_loss += F.mse_loss(gen_feat, target_feat)
    # calculate total style loss
    for gen_feat, target_feat in zip(generated_style, style_targets):
        s_loss += style_loss(gen_feat, target_feat)

    # combine the losses (perceptual loss)
    loss = content_weight * c_loss + style_weight * s_loss
    return {
        "perceptual": loss, "content": c_loss, "style": s_loss
    }
