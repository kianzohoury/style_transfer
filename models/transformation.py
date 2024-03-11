
from typing import Optional

import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Simple convolutional block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding_mode: str = "reflect",
        norm_type: Optional[str] = "instance",
        has_activation: bool = True
    ):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.norm_type = norm_type
        self.padding_size = kernel_size // 2

        if norm_type == "instance":
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm2d(out_channels, affine=True)
        else:
            self.norm = nn.Identity()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding_size,
            padding_mode=padding_mode,
            bias=True if norm_type is None else False
        )
        self.relu = nn.ReLU() if has_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class TransposeConv(ConvBlock):
    """Transpose/deconvolution block."""
    def __init__(self, *args, **kwargs):
        super(TransposeConv, self).__init__(*args, **kwargs)
        output_padding = self.kernel_size // 2
        self.conv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.padding_size,
            output_padding=output_padding,
            bias=True if self.norm_type is None else False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class UpsampleConv(ConvBlock):
    """Performs upsampling via interpolation (instead of transpose conv)."""
    def __init__(self, *args, **kwargs):
        super(UpsampleConv, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            input=x, scale_factor=2, mode="nearest-exact"
        )
        x = self.forward(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block."""
    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        padding_mode: str = "reflect"
    ):
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding_mode=padding_mode
        )
        self.conv2 = ConvBlock(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            has_activation=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + skip


class TransformationNetwork(nn.Module):
    """Style Transformation network proposed in Johnson et al."""
    def __init__(
        self,
        padding_mode: str = "reflect",
        norm_type: str = "instance",
        upsample_type: str = "transpose",
        output_fn: Optional[str] = None
    ):
        super(TransformationNetwork, self).__init__()

        # encoder blocks
        self.conv_in = ConvBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=9,
            padding_mode=padding_mode,
            norm_type=norm_type
        )
        self.down1 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding_mode=padding_mode,
            norm_type=norm_type
        )
        self.down2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding_mode=padding_mode,
            norm_type=norm_type
        )

        # residual bottleneck
        self.residual = nn.Sequential(
            ResidualBlock(
                num_channels=128, kernel_size=3, padding_mode=padding_mode
            ),
            ResidualBlock(
                num_channels=128, kernel_size=3, padding_mode=padding_mode
            ),
            ResidualBlock(
                num_channels=128, kernel_size=3, padding_mode=padding_mode
            ),
            ResidualBlock(
                num_channels=128, kernel_size=3, padding_mode=padding_mode
            ),
            ResidualBlock(
                num_channels=128, kernel_size=3, padding_mode=padding_mode
            )
        )

        if upsample_type == "transpose":
            upsampler = TransposeConv
        elif upsample_type == "interpolate":
            upsampler = UpsampleConv
        else:
            raise ValueError(
                f"{upsample_type} is not a valid upsampling type."
            )

        # decoder blocks
        self.up1 = upsampler(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2 if upsample_type == "transpose" else 1,
            padding_mode=padding_mode,
            norm_type=norm_type
        )
        self.up2 = upsampler(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2 if upsample_type == "transpose" else 1,
            padding_mode=padding_mode,
            norm_type=norm_type
        )
        self.conv_out = ConvBlock(
            in_channels=32,
            out_channels=3,
            kernel_size=9,
            padding_mode=padding_mode,
            norm_type=None,
            has_activation=False
        )

        # optionally squeeze output values using clamp or sigmoid
        if output_fn == "clamp":
            self.output_fn = lambda x: torch.clamp(x, 0, 1)
        elif output_fn == "sigmoid":
            self.output_fn = nn.Sigmoid()
        else:
            self.output_fn = nn.Identity()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(image)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        residual = self.residual(d2)
        up1 = self.up1(residual)
        up2 = self.up2(up1)
        out = self.conv_out(up2)
        return self.output_fn(out)

#
# class TransformationNetworkSkip(nn.Module):
#     """Style Transformation network proposed in Johnson et al."""
#     def __init__(self):
#         super(TransformationNetworkSkip, self).__init__()
#         self.conv_in = nn.Sequential(
#             nn.Conv2d(3, 32, 9, padding="same", padding_mode="reflect", bias=False),
#             # nn.BatchNorm2d(32),
#             nn.InstanceNorm2d(32, affine=True),
#             nn.ReLU(),
#         )
#         self.down_1 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode="reflect", bias=False),
#             # nn.BatchNorm2d(64),
#             nn.InstanceNorm2d(64, affine=True),
#             nn.ReLU()
#         )
#         self.down_2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode="reflect", bias=False),
#             # nn.BatchNorm2d(128),
#             nn.InstanceNorm2d(128, affine=True),
#             nn.ReLU()
#         )
#         self.residual = nn.Sequential(
#             ResidualBlock(num_channels=128, dilation=1),
#             ResidualBlock(num_channels=128, dilation=1),
#             ResidualBlock(num_channels=128, dilation=1),
#             ResidualBlock(num_channels=128, dilation=1),
#             ResidualBlock(num_channels=128, dilation=1)
#         )
#         self.up_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, bias=False)
#         self.act_1 = nn.Sequential(
#             # nn.BatchNorm2d(64),
#             nn.InstanceNorm2d(64, affine=True),
#             nn.ReLU()
#         )
#         self.up_1_conv = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding="same", padding_mode="reflect", bias=False),
#             # nn.BatchNorm2d(32),
#             nn.InstanceNorm2d(64, affine=True),
#             nn.ReLU()
#         )
#         self.up_2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, bias=False)
#         self.act_2 = nn.Sequential(
#             # nn.BatchNorm2d(32),
#             nn.InstanceNorm2d(32, affine=True),
#             nn.ReLU()
#         )
#         self.up_2_conv = nn.Sequential(
#             nn.Conv2d(64, 32, 3, padding="same", padding_mode="reflect", bias=False),
#             # nn.BatchNorm2d(32),
#             nn.InstanceNorm2d(32, affine=True),
#             nn.ReLU()
#         )
#         conv_out = nn.Conv2d(32, 3, 9, padding="same", padding_mode="reflect")
#         self.conv_out = nn.Sequential(
#             conv_out,
#             # nn.InstanceNorm2d(3, affine=True),
#         )
#         # nn.init.xavier_normal_(conv_out.weight, gain=1.0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         x = self.conv_in(image) # 32 x 256 256
#         down_1 = self.down_1(x) # 64 x 128 x 128
#         down_2 = self.down_2(down_1) # 128 x 64 x 64
#         residual = self.residual(down_2)
#         up_1 = self.act_1(self.up_1(residual, output_size=down_1.size()))
#         up_1 = self.up_1_conv(torch.cat([up_1, down_1], dim=1))
#         up_2 = self.act_2(self.up_2(up_1, output_size=x.size()))
#         up_2 = self.up_2_conv(torch.cat([up_2, x], dim=1))
#         out = self.conv_out(up_2)
#         # out = self.sigmoid(out)
#         return out
