
from transformation import TransformationNetwork
from vgg import VGGNetwork

import torch.backends.cudnn

__all__ = [TransformationNetwork, VGGNetwork]

# Use CNN GPU optimizations if available.
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
