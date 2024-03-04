
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import torch.backends.cudnn
import torch.nn as nn
import torchvision

# Note that we label the layers conv_i_j instead of relu_i_j, but these
# are equivalent, as the features are still extracted after the activation
# layer.
_VALID_LAYERS_16 = [
    "conv_1_1",
    "conv_1_2",
    "conv_2_1",
    "conv_2_2",
    "conv_3_1",
    "conv_3_2",
    "conv_3_3",
    "conv_4_1",
    "conv_4_2",
    "conv_4_3",
    "conv_5_1",
    "conv_5_2",
    "conv_5_3"
]
_VALID_LAYERS_19 = sorted(_VALID_LAYERS_16 + ["conv_4_4", "conv_5_4"])
DEFAULT_STYLE_LAYERS_16 = [
    "conv_1_2", "conv_2_2", "conv_3_3", "conv_4_3"
]
DEFAULT_STYLE_LAYERS_19 = [
    "conv_1_2", "conv_2_2", "conv_3_3", "conv_4_4", "conv_5_4"
]
DEFAULT_CONTENT_LAYERS = ["conv_4_2"]


class VGGNetwork(nn.Module):
    """Pretrained VGG network that extracts feature representations.

    Feature representations from content/style images are extracted from this
    network to calculate their perceptual losses for optimization. During
    the forward pass, the desired feature maps are stored for later access.

    Args:
        model_type: {vgg16, vgg19, vgg16_bn, vgg19_bn}. Default: "vgg16".
        use_avg_pool (bool): Whether to use average pooling instead of max
            pooling (recommended by Gatys et al.). Default: True.
        feature_labels (list, optional): VGG layers to extract features from.
            If None are specified, a default set of layers (see Gatys et al.,
            Johnson et al.) will be used.
    """
    _feature_maps: Dict[str, torch.Tensor]

    def __init__(
        self,
        model_type: str = "vgg16",
        use_avg_pool: bool = True,
        feature_labels: Optional[List[str]] = None
    ):
        super(VGGNetwork, self).__init__()

        valid_models = ["vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
        if model_type.lower() not in valid_models:
            raise ValueError(f"Unknown model. Must be one of {valid_models}.")
        self.model_type = model_type.lower()
        self.use_avg_pool = use_avg_pool

        # Load pretrained weights
        weights_to_load = getattr(
            torchvision.models, self.model_type.upper() + "_Weights"
        ).IMAGENET1K_V1
        features = getattr(torchvision.models, self.model_type)(
            weights=weights_to_load
        ).features

        # check if specified layers are valid
        is_valid_layers = True
        if feature_labels is not None:
            if "16" in self.model_type and not \
                    set(feature_labels) <= set(_VALID_LAYERS_16):
                is_valid_layers = False
            elif "19" in self.model_type and not \
                    set(feature_labels) <= set(_VALID_LAYERS_19):
                is_valid_layers = False
            if not is_valid_layers:
                raise ValueError(f"{feature_labels} are not valid layers.")
            self.feature_labels = sorted(list(set(feature_labels)))
        else:
            labels = DEFAULT_CONTENT_LAYERS[:]
            if "16" in self.model_type:
                labels += DEFAULT_STYLE_LAYERS_16
            else:
                labels += DEFAULT_STYLE_LAYERS_19
            self.feature_labels = sorted(labels)

        # name each layer
        named_layers = []
        block_i, sublayer_j = 1, 0
        for layer in list(features.children()):
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
                if self.use_avg_pool:
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

        # build network
        self.network = nn.Sequential()
        self._feature_maps = {}

        i = 0
        while i < len(named_layers):
            label, layer = named_layers[i]
            if label not in self.feature_labels:
                # add the current layer
                self.network.add_module(label, layer)
                i += 1
            else:
                j = i + 1

                # Include all layers up until the next conv/pool layer
                while j < len(named_layers):
                    next_layer_type = named_layers[j][0].split("_")[0]
                    if next_layer_type not in {"conv", "pool"}:
                        j += 1
                    else:
                        break

                feature_extraction_layers = nn.Sequential(
                    OrderedDict(named_layers[i:j])
                )
                feature_extraction_layers.register_forward_hook(
                    get_activation(label, self._feature_maps)
                )
                # add layers so far to the network
                self.network.add_module(
                    label, feature_extraction_layers
                )
                i = j

            # stop building network after final feature layer is added
            if label == self.feature_labels[-1]:
                break

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs forward pass and returns feature representations."""
        self._feature_maps = {}
        self.network(image)
        return self._feature_maps


def get_activation(
    name: str, stored_activations: Dict[str, torch.Tensor]
) -> Callable:
    """Returns a forward hook that stores the specified activations."""
    def hook(module, x, output):
        stored_activations[name] = output
    return hook
