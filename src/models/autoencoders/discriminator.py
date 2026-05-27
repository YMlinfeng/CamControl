from dataclasses import dataclass, field
from typing import Any, Tuple, Type, Union

import torch
from einops import rearrange
from torch import einsum, nn

try:
    from kornia.filters import filter3d
except ImportError:
    Warning("kornia not installed")

from ...configs.base_config import InstantiateConfig
from ...utils import load_model, log_to_rank0


class BlurPool(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])  # 1d kernel
        self.register_buffer("f", f)
        self.avgpool3d = nn.AvgPool3d(2, 2)
        self.avgpool2d = nn.AvgPool2d(2, 2)

    def forward(self, x):
        f = self.f
        f = einsum("i, j, k -> i j k", f, f, f)
        f = rearrange(f, "... -> 1 ...")

        is_images = x.shape[2] == 1

        dtype = x.dtype
        x = x.float()
        f = f.float()
        out = filter3d(x, f, normalized=True)
        out = out.type(dtype)

        if is_images:
            out = rearrange(out, "b c 1 h w -> b c h w")
            out = self.avgpool2d(out)
            out = rearrange(out, "b c h w -> b c 1 h w")
        else:
            out = self.avgpool3d(out)

        return out


class ResBlockDownX2Y(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(chan_in, chan_out, kernel_size=kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2),
            BlurPool(),
            nn.Conv3d(chan_out, chan_out, kernel_size=kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.skip_net = nn.Sequential(BlurPool(), nn.Conv3d(chan_in, chan_out, kernel_size=(1, 1, 1), stride=stride))

    def forward(self, x):
        c = x
        x = self.net(x)
        skip = self.skip_net(c)

        return x + skip


@dataclass
class DiscriminatorConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Discriminator)

    channels: int = 3
    init_dim: int = 128
    kernel_size: Tuple[int, int, int] = (3, 3, 3)
    use_pixel_discriminator: bool = False

    def from_pretrained(self, ckpt_path=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        discriminator = self._target(self)
        if ckpt_path is not None:
            discriminator = load_model(discriminator, ckpt_path)
        return discriminator


class Discriminator(nn.Module):
    def __init__(
        self,
        config: DiscriminatorConfig,
    ):
        super().__init__()

        self.config = config
        channels = config.channels
        init_dim = config.init_dim
        kernel_size = config.kernel_size

        self.conv_in = nn.Sequential(
            nn.Conv3d(channels, init_dim, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.resblockdown1 = ResBlockDownX2Y(init_dim, init_dim * 2)
        self.resblockdown2 = ResBlockDownX2Y(init_dim * 2, init_dim * 4)
        self.resblockdown3 = nn.ModuleList([ResBlockDownX2Y(init_dim * 4, init_dim * 4) for _ in range(3)])

        self.conv_out = nn.Sequential(
            nn.Conv3d(init_dim * 4, init_dim * 4, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.linear = nn.Sequential(nn.Linear(init_dim * 4, 512), nn.LeakyReLU(0.2), nn.Linear(512, 1))

        if config.use_pixel_discriminator:
            self.pixel_discriminator = PixelDiscriminator(channels, init_dim, num_groups=32)

    def forward(self, x, return_features=False):
        features = []
        x = self.conv_in(x)
        x = self.resblockdown1(x)
        features.append(x)
        x = self.resblockdown2(x)
        features.append(x)
        for i in range(3):
            x = self.resblockdown3[i](x)
            features.append(x)
        x = self.conv_out(x)
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f h w) c")
        x = self.linear(x)
        x = rearrange(x, "(b f h w) 1 -> b f h w", b=b, f=f, h=h, w=w)

        if return_features:
            return x, features
        return x


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, channels, init_dim=64, num_groups=32):

        super(PixelDiscriminator, self).__init__()

        self.net = [
            nn.Conv3d(channels, init_dim, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(init_dim, init_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=num_groups, num_channels=init_dim * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(init_dim * 2, 1, kernel_size=1, stride=1, padding=0),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


# input = torch.randn(2, 16, 3, 128, 128).to("cuda")

# dis = Discriminator().to("cuda")
# out = dis(input)
# print(out.shape)
