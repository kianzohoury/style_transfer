import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, List, Optional

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

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

    content_layer: List[LossLayer]
    style_layer = List[LossLayer]
    
    def __init__(
            self,
            model: nn.Module,
            content_image: torch.Tensor,
            style_image: torch.Tensor,
            content_labels: List[str],
            style_labels: List[str],
    ):
        super(VGGLossNet, self).__init__()
        self.content_layers, self.style_layers = [], []
        self.network = nn.Sequential()

        # store constructor args 
        self.content_image = content_image.clone()
        self.style_image = style_image.clone()
        self.content_labels = content_labels[:]
        self.style_labels = style_labels[:]

        self.network.add_module(
            "input_norm",
            InputNorm(
                mean=_IMAGENET_MEAN,
                std=_IMAGENET_STD
            ).to(content_image.device)
        )
        
        block_i, sublayer_j = 1, 0
        named_layers = []
        for layer in list(model.children()):
            layer.requires_grad = False

            if isinstance(layer, nn.Conv2d):
                sublayer_j += 1
                label = f"conv_{block_i}_{sublayer_j}"
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
                sublayer_j = 0
            else:
                label = layer.__class__.__name__
            named_layers.append((label, layer))
        
        i = 0
        loss_labels = set(self.content_labels) | set(self.style_labels)
        while i < len(named_layers):
            label, layer = named_layers[i]
            if label not in loss_labels:
                # add the current layer
                self.network.add_module(label, layer)
                i += 1
            else:
                j = i + 1

                # include all activation layers following the content/style loss layer
                # up until the next convolutional or pooling layer
                while j < len(named_layers) and named_layers[j][0].split("_")[0] not in ["conv", "pool"]:
                    j += 1

                # add layers so far to the network
                for named_layer in named_layers[i:j]:
                    self.network.add_module(*named_layer)

                if self.content_labels and label == self.content_labels[0]:
                    content_label = self.content_labels.pop(0)
                    # add content loss layer
                    content_features = self.network(content_image).detach()
                    content_layer = LossLayer(
                        target_features=content_features, loss_fn=content_loss
                    )
                    self.network.add_module(
                        f"content_{content_label}", content_layer
                    )
                    self.content_layers.append(content_layer)
                if self.style_labels and label == self.style_labels[0]:
                    style_label = self.style_labels.pop(0)
                    # add style loss layer
                    style_features = self.network(style_image).detach()
                    style_layer = LossLayer(
                        target_features=style_features, loss_fn=style_loss
                    )
                    self.network.add_module(
                        f"style_{style_label}", style_layer
                    )
                    self.style_layers.append(style_layer)
                i = j

            # stop building network after final content/style loss layer is added
            if not (self.content_labels or self.style_labels):
                break

        if self.content_labels or self.style_labels:
            invalid_labels = set(self.content_labels) | set(self.style_labels)
            raise ValueError(f"{invalid_labels} are not valid layers.")
        self.content_labels = content_labels[:]
        self.style_labels = style_labels[:]

    def forward(self, image: torch.Tensor) -> List[List[LossLayer]]:
        """Returns the feature maps resulting from the final layer in the loss network."""
        return self.network(image)


class ResidualBlock(nn.Module):
    """Residual block scheme for `TransformationNetwork`."""
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding="same", bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding="same", bias=False),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features) + features


class TransformationNetwork(nn.Module):
    """Style Transformation network proposed in Johnson et al."""
    def __init__(self):
        super(TransformationNetwork, self).__init__()
        self.input_norm = InputNorm(
            mean=_IMAGENET_MEAN, std=_IMAGENET_STD
        )
        self.conv_in = nn.Sequential(
            nn.Conv2d(3, 32, 9, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.down_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.down_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            ResidualBlock(num_channels=128),
            ResidualBlock(num_channels=128),
            ResidualBlock(num_channels=128),
            ResidualBlock(num_channels=128),
            ResidualBlock(num_channels=128)
        )
        self.up_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, bias=False)
        self.act_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up_2 = nn.ConvTranspose2d(64, 32, 3, stride=2, bias=False)
        self.act_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_out = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 9, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(image)
        x = self.conv_in(x)
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        residual = self.residual(down_2)
        up_1 = self.act_1(self.up_1(residual, output_size=down_1.size()))
        up_2 = self.act_2(self.up_2(up_1, output_size=x.size()))
        out = self.sigmoid(up_2)
        return out


def gram_matrix(feature_maps: torch.Tensor) -> torch.Tensor:
    """Returns the normalized Gram matrix of the input features."""
    n, c, h, w = feature_maps.shape
    feature_maps = feature_maps.reshape((n * c, h * w))
    gram = feature_maps @ feature_maps.T
    return gram / (n * c * h * w)


def content_loss(
    generated_features: torch.Tensor,
    target_features: torch.Tensor
) -> torch.Tensor:
    """Returns the content loss given the generated and target features."""
    loss = F.mse_loss(generated_features, target_features)
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


def total_variation_loss(feature_maps: torch.Tensor) -> torch.Tensor:
    """Returns total variation (TV) loss of the given input features.

    Implements isotropic and differentiable version of TV loss, without
    using square root to aggregate squared differences.
    """
    # sum right neighbor pixel differences
    loss = torch.sum(
        (feature_maps[:, :, :, :-1] - feature_maps[:, :, :, 1:]) ** 2
    )
    # sum lower neighbor pixel differences
    loss += torch.sum(
        (feature_maps[:, :, :-1, :] - feature_maps[:, :, 1:, :]) ** 2
    )
    # return the mean loss
    return loss / feature_maps.numel()


def perceptual_loss(
    content_losses: List[LossLayer],
    style_losses: List[LossLayer],
    content_weight: float,
    style_weight: float,
    generated_img: Optional[torch.Tensor] = None,
    tv_weight: float = 0
):
    """Calculates the perceptual loss, which combines content and style losses.

    Optionally, total variation regularization is applied if the generated image is passed in.
    Returns the combined loss, followed by content, style, and TV losses.
    """
    c_loss = s_loss = tv_loss = 0
    # calculate total content loss
    for loss_layer in content_losses:
        c_loss += loss_layer.loss
    # calculate total style loss
    for loss_layer in style_losses:
        s_loss += loss_layer.loss

    # combine the losses (perceptual loss)
    loss = content_weight * c_loss + style_weight * s_loss

    # additionally calculate TV loss
    if generated_img is not None:
        tv_loss += tv_weight * total_variation_loss(generated_img)
        loss += tv_loss
    return [loss, c_loss, s_loss, tv_loss]
