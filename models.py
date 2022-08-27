

import torch
import torch.nn as nn

from typing import Callable, List


class InputNorm(nn.Module):
    """Normalizes input given the mean and std."""
    def __init__(self, mean: List[float], std: List[float]):
        super(InputNorm, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).view(-1, 1, 1))
        self.std = nn.Parameter(torch.Tensor(std).view(-1, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = (image - self.mean) / self.std
        return normalized


class LossLayer(nn.Module):
    """Stores the layer's loss given the target feature maps and loss fn."""
    loss: torch.Tensor
    target_features: torch.Tensor

    def __init__(self, target_features: torch.Tensor, loss_fn: Callable):
        super(LossLayer, self).__init__()
        self.target_features = target_features
        self.loss_fn = loss_fn

    def forward(self, generated_features: torch.Tensor) -> torch.Tensor:
        self.loss = self.loss_fn(generated_features, self.target_features)
        return generated_features


class StyleTransferNet(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            content_image: torch.Tensor,
            style_image: torch.Tensor,
            content_labels: List[str],
            style_labels: List[str],
            mean: List[float],
            std: List[float]
    ):
        super(StyleTransferNet, self).__init__()
        self.input_size = content_image.size()
        self.network = nn.Sequential()
        self.network.add_module(
            "input_norm",
            InputNorm(mean=mean, std=std).to(content_image.device)
        )
        self.content_layers = []
        self.style_layers = []

        i = 1
        j = 0

        for layer in list(model.children()):
            layer.requires_grad = False
            if isinstance(layer, nn.Conv2d):
                j += 1
                label = f"conv_{i}_{j}"
            elif isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
                label = f"relu_{i}_{j}"
            elif isinstance(layer, nn.BatchNorm2d):
                label = f"bn_{i}_{j}"
            elif isinstance(layer, nn.MaxPool2d):
                # Use avg pooling instead.
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                layer = nn.AvgPool2d(kernel_size, stride, padding)
                label = f"pool_{i}_{j}"
                i += 1
                j = 0
            else:
                label = layer.__class__.__name__

            self.network.add_module(label, layer)
            if content_labels and label == content_labels[0]:
                content_labels.pop(0)
                content_features = self.network(content_image).detach()
                content_layer = LossLayer(
                    target_features=content_features, loss_fn=content_loss
                )
                self.content_layers.append(content_layer)
                self.network.add_module(
                    f"content_{len(self.content_layers) + 1}", content_layer
                )
            if style_labels and label == style_labels[0]:
                style_labels.pop(0)
                style_features = self.network(style_image).detach()
                style_layer = LossLayer(
                    target_features=style_features, loss_fn=style_loss
                )
                self.style_layers.append(style_layer)
                self.network.add_module(
                    f"style_{len(self.style_layers) + 1}", style_layer
                )
            if not content_labels and not style_labels:
                break

    def forward(self, image: torch.Tensor) -> List[List[LossLayer]]:
        """Returns the content and style layers as a tuple of lists."""
        _ = self.network(image)
        return [self.content_layers, self.style_layers]


def gram_matrix(feature_maps: torch.Tensor) -> torch.Tensor:
    """Returns normalized Gram matrix of the input features."""
    n, c, h, w = feature_maps.shape
    feature_maps = feature_maps.reshape((n * c, h * w))
    gram = feature_maps @ feature_maps.T
    return gram / (n * c * h * w)


def content_loss(
    generated_features: torch.Tensor,
    target_features: torch.Tensor
) -> torch.Tensor:
    """Returns the content loss given the generated and target features."""
    loss = nn.functional.mse_loss(generated_features, target_features)
    return loss


def style_loss(
    generated_features: torch.Tensor,
    target_features: torch.Tensor
) -> torch.Tensor:
    """Returns the style loss given the generated and target features."""
    generated_gram = gram_matrix(generated_features)
    target_gram = gram_matrix(target_features)
    loss = nn.functional.mse_loss(generated_gram, target_gram)
    return loss


def total_variation_loss(feature_maps: torch.Tensor) -> torch.Tensor:
    """Returns the total variation loss of the input features."""
    loss = torch.sum(
        feature_maps[:, :, :, :-1] - feature_maps[:, :, :, 1:] ** 2
    )
    loss += torch.sum(
        feature_maps[:, :, :-1, :] + feature_maps[:, :, 1:, :] ** 2
    )
    return loss
