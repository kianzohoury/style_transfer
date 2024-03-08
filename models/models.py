
import torch
import torch.backends.cudnn
import torch.nn as nn


from typing import Callable, List

# Use CNN GPU optimizations if available.
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True


class InputNorm(nn.Module):
    """Normalizes input given a mean and std."""
    def __init__(self, mean: List[float], std: List[float]):
        super(InputNorm, self).__init__()
        self.mean = nn.Parameter(
            torch.Tensor(mean).view(-1, 1, 1), requires_grad=False
        )
        self.std = nn.Parameter(
            torch.Tensor(std).view(-1, 1, 1), requires_grad=False
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = (image - self.mean) / self.std
        return normalized


class LossLayer(nn.Module):
    """Stores the layer's loss given the target feature maps and loss fn.

    Returns the unmodified input features (identity function) so they can be fed into
    the next layer.
    """
    loss: torch.Tensor
    target_features: torch.Tensor

    def __init__(self, target_features: torch.Tensor, loss_fn: Callable):
        super(LossLayer, self).__init__()
        self.target_features = target_features
        self.loss_fn = loss_fn

    def forward(self, generated_features: torch.Tensor) -> torch.Tensor:
        self.loss = self.loss_fn(generated_features, self.target_features)
        return generated_features


class VGGLossNet(nn.Module):
    """Network that computes the content and style losses."""
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
        super(VGGLossNet, self).__init__()
        self.network = nn.Sequential()

        # store constructor args for loading model during inference
        self.content_image = content_image.clone()
        self.style_image = style_image.clone()
        self.content_labels = content_labels[:]
        self.style_labels = style_labels[:]
        self.mean = mean
        self.std = std

        self.network.add_module(
            "input_norm",
            InputNorm(mean=mean, std=std).to(content_image.device)
        )

        self.content_layers, self.style_layers = [], []
        block_i = sublayer_j = 1
        prev = curr = None

        for layer in list(model.children()):
            layer.requires_grad = False

            if isinstance(layer, nn.Conv2d):
                label = f"conv_{block_i}_{sublayer_j}"
                sublayer_j += 1

                if (content_labels and label == content_labels[0]) \
                    or (style_labels and label == style_labels[0]):
                    prev = label
                else:
                    curr = label
            elif isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
                label = f"relu_{block_i}_{sublayer_j}"
            elif isinstance(layer, nn.BatchNorm2d):
                label = f"bn_{block_i}_{sublayer_j}"
            elif isinstance(layer, nn.MaxPool2d):
                # Use avg pooling instead.
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                layer = nn.AvgPool2d(kernel_size, stride, padding)
                label = f"pool_{block_i}_{sublayer_j}"
                block_i += 1
                sublayer_j = 1
            else:
                label = layer.__class__.__name__

            self.network.add_module(label, layer)
            if prev and "conv" in curr:
                if content_labels:
                    content_labels.pop(0)
                    content_features = self.network(content_image).detach()
                    content_layer = LossLayer(
                        target_features=content_features, loss_fn=content_loss
                    )
                    self.content_layers.append(content_layer)
                    self.network.add_module(
                        f"content_{len(self.content_layers)}", content_layer
                    )
                if style_labels:
                    style_labels.pop(0)
                    style_features = self.network(style_image).detach()
                    style_layer = LossLayer(
                        target_features=style_features, loss_fn=style_loss
                    )
                    self.style_layers.append(style_layer)
                    self.network.add_module(
                        f"style_{len(self.style_layers)}", style_layer
                    )
            if not content_labels and not style_labels:
                break

    def forward(self, image: torch.Tensor) -> List[List[LossLayer]]:
        """Returns the feature maps resulting from the last layer in the network."""
        return self.network(image)


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
    """Returns total variation (TV) loss of the given input features.

    Implements isotropic and differentiable version of TV loss, without
    using square root to aggregate squared differences.
    """
    # sum right neighbor pixel differences
    loss = (feature_maps[:, :, :, :-1] - feature_maps[:, :, :, 1:]) ** 2
    # sum lower neighbor pixel differences
    loss += (feature_maps[:, :, :-1, :] - feature_maps[:, :, 1:, :]) ** 2
    # return the mean loss
    return loss / feature_maps.numel()
