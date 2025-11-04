import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm


def is_tuple_of(obj: Any, expected_type: type) -> bool:
    return isinstance(obj, tuple) and all(
        isinstance(item, expected_type) for item in obj
    )


def build_activation_layer(cfg: Dict[str, Any]) -> nn.Module:
    cfg = cfg.copy()
    act_type = cfg.pop("type")
    if act_type == "ReLU":
        return nn.ReLU(**cfg)
    if act_type == "LeakyReLU":
        return nn.LeakyReLU(**cfg)
    if act_type == "Sigmoid":
        return nn.Sigmoid()
    if act_type == "SiLU":
        return nn.SiLU(**cfg)
    if act_type == "GELU":
        return nn.GELU(**cfg)
    if act_type == "Identity":
        return nn.Identity()
    raise NotImplementedError(f"Activation type '{act_type}' is not supported.")


def build_conv_layer(
    conv_cfg: Optional[Dict[str, Any]],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: Optional[bool] = None,
    **kwargs: Any,
) -> nn.Module:
    cfg = (conv_cfg or {}).copy()
    conv_type = cfg.pop("type", "Conv2d")
    if bias is None:
        bias = cfg.pop("bias", True)
    else:
        cfg.pop("bias", None)

    if conv_type != "Conv2d":
        raise NotImplementedError(f"Conv layer type '{conv_type}' is not supported.")

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        **cfg,
        **kwargs,
    )


def build_norm_layer(
    norm_cfg: Dict[str, Any], num_features: int
) -> Tuple[str, nn.Module]:
    cfg = (norm_cfg or {}).copy()
    norm_type = cfg.pop("type", "BN")
    if norm_type in {"BN", "SyncBN"}:
        layer = nn.BatchNorm2d(num_features, **cfg)
        name = "bn"
    elif norm_type == "GN":
        num_groups = cfg.pop("num_groups")
        layer = nn.GroupNorm(num_groups, num_features, **cfg)
        name = "gn"
    elif norm_type == "LN":
        layer = nn.LayerNorm(num_features, **cfg)
        name = "ln"
    else:
        raise NotImplementedError(f"Normalization type '{norm_type}' is not supported.")
    return name, layer


def constant_init(module: nn.Module, val: float, bias: float = 0.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean: float = 0.0, std: float = 1.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.view(batch_size, -1, height, width)


def get_root_logger(name: str = "LiteHRNet") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_checkpoint(
    model: nn.Module,
    filename: str,
    strict: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(filename, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if logger is None:
        logger = get_root_logger()
    if missing_keys:
        logger.warning("Missing keys while loading checkpoint: %s", missing_keys)
    if unexpected_keys:
        logger.warning("Unexpected keys while loading checkpoint: %s", unexpected_keys)
    return checkpoint


class ConvModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Dict[str, Any]] = None,
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Optional[Dict[str, Any]] = dict(type="ReLU"),
    ) -> None:
        super().__init__()
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if self.with_norm:
            norm_channels = out_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
                "GELU",
            ]:
                act_cfg_.setdefault("inplace", True)
            self.activate = build_activation_layer(act_cfg_)

    @property
    def norm(self) -> Optional[nn.Module]:
        if self.norm_name is not None:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.with_norm:
            x = self.norm(x)
        if self.with_activation:
            x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        conv_cfg: Optional[Dict[str, Any]] = None,
        norm_cfg: Optional[Dict[str, Any]] = None,
        act_cfg: Optional[Dict[str, Any]] = dict(type="ReLU"),
        dw_norm_cfg: Optional[Dict[str, Any]] = None,
        pw_norm_cfg: Optional[Dict[str, Any]] = None,
        dw_act_cfg: Optional[Dict[str, Any]] = None,
        pw_act_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg is not None else norm_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg is not None else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg is not None else act_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg is not None else act_cfg

        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            conv_cfg=conv_cfg,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
        )
        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SpatialWeighting(nn.Module):

    def __init__(
        self,
        channels,
        ratio=16,
        conv_cfg=None,
        act_cfg=(dict(type="ReLU"), dict(type="Sigmoid")),
    ):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0],
        )
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1],
        )

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class CrossResolutionWeighting(nn.Module):

    def __init__(
        self,
        channels,
        ratio=16,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=(dict(type="ReLU"), dict(type="Sigmoid")),
    ):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0],
        )
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1],
        )

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode="nearest")
            for s, a in zip(x, out)
        ]
        return out


class ConditionalChannelWeighting(nn.Module):

    def __init__(
        self,
        in_channels,
        stride,
        reduce_ratio,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels, ratio=reduce_ratio, conv_cfg=conv_cfg, norm_cfg=norm_cfg
        )

        self.depthwise_convs = nn.ModuleList(
            [
                ConvModule(
                    channel,
                    channel,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=channel,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                )
                for channel in branch_channels
            ]
        )

        self.spatial_weighting = nn.ModuleList(
            [SpatialWeighting(channels=channel, ratio=4) for channel in branch_channels]
        )

    def forward(self, x):

        def _inner_forward(x):
            x = [s.chunk(2, dim=1) for s in x]
            x1 = [s[0] for s in x]
            x2 = [s[1] for s in x]

            x2 = self.cross_resolution_weighting(x2)
            x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
            x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

            out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
            out = [channel_shuffle(s, 2) for s in out]

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class Stem(nn.Module):

    def __init__(
        self,
        in_channels,
        stem_channels,
        out_channels,
        expand_ratio,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type="ReLU"),
        )

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=branch_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type="ReLU"),
            ),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="ReLU"),
        )
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="ReLU"),
        )

    def forward(self, x):

        def _inner_forward(x):
            x = self.conv1(x)
            x1, x2 = x.chunk(2, dim=1)

            x2 = self.expand_conv(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.linear_conv(x2)

            out = torch.cat((self.branch1(x1), x2), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class IterativeHead(nn.Module):

    def __init__(self, in_channels, conv_cfg=None, norm_cfg=dict(type="BN")):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type="ReLU"),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type="ReLU"),
                    )
                )
            else:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type="ReLU"),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type="ReLU"),
                    )
                )
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]

        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x, size=s.size()[-2:], mode="bilinear", align_corners=True
                )
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]


class ConcatHead(nn.Module):
    def __init__(
        self, in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type="BN")
    ):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=sum(in_channels),
                out_channels=sum(in_channels),
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type="ReLU"),
                dw_act_cfg=None,
                pw_act_cfg=dict(type="ReLU"),
            ),
            nn.Conv2d(
                sum(in_channels),
                out_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x):
        size = x[0].shape[2:]
        x = torch.cat(
            [
                F.interpolate(s, size=size, mode="bilinear", align_corners=True)
                for s in x
            ],
            dim=1,
        )
        x = self.conv(x)
        return x


class FaceAlignmentHeatmapHead(nn.Module):
    def __init__(
        self, in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type="BN")
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.heatmap_layer = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels=in_channels[0],
                out_channels=in_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type="ReLU"),
                dw_act_cfg=None,
                pw_act_cfg=dict(type="ReLU"),
            ),
            nn.Conv2d(
                in_channels=in_channels[0],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        x = x[0]
        x = self.heatmap_layer(x)
        return x


class ShuffleUnit(nn.Module):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        with_cp=False,
    ):
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f"in_channels ({in_channels}) should equal to "
                f"branch_features * 2 ({branch_features * 2}) "
                "when stride is 1"
            )

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f"stride ({self.stride}) should not equal 1 when "
                f"in_channels != branch_features * 2"
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                ),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

    def forward(self, x):

        def _inner_forward(x):
            if self.stride > 1:
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class LiteHRModule(nn.Module):

    def __init__(
        self,
        num_branches,
        num_blocks,
        in_channels,
        reduce_ratio,
        module_type,
        multiscale_output=False,
        with_fuse=True,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp

        if self.module_type == "LITE":
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == "NAIVE":
            self.layers = self._make_naive_branches(num_branches, num_blocks)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = (
                f"NUM_BRANCHES({num_branches}) "
                f"!= NUM_INCHANNELS({len(in_channels)})"
            )
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp,
                )
            )

        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        """Make one branch."""
        layers = []
        layers.append(
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=stride,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type="ReLU"),
                with_cp=self.with_cp,
            )
        )
        for i in range(1, num_blocks):
            layers.append(
                ShuffleUnit(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=dict(type="ReLU"),
                    with_cp=self.with_cp,
                )
            )

        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[i])[1],
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        if self.module_type == "LITE":
            out = self.layers(x)
        elif self.module_type == "NAIVE":
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]
        return out


class LiteHRNet(nn.Module):
    """Lite-HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`_

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmpose.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)
    """

    def __init__(
        self,
        extra,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
        **kwargs,
    ):
        super().__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.downsample_ratio = 32

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra["stem"]["stem_channels"],
            out_channels=self.extra["stem"]["out_channels"],
            expand_ratio=self.extra["stem"]["expand_ratio"],
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
        )

        self.num_stages = self.extra["num_stages"]
        self.stages_spec = self.extra["stages_spec"]

        num_channels_last = [
            self.stem.out_channels,
        ]
        for i in range(self.num_stages):
            num_channels = self.stages_spec["num_channels"][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            setattr(
                self,
                "transition{}".format(i),
                self._make_transition_layer(num_channels_last, num_channels),
            )

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, multiscale_output=True
            )
            setattr(self, "stage{}".format(i), stage)

        self.with_head = self.extra["with_head"]
        if self.with_head:
            self.head_layer = IterativeHead(
                in_channels=num_channels_last,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
            )

        final_inp_channels = sum(num_channels_last)
        self.num_out_feats = [
            final_inp_channels,
            final_inp_channels,
            final_inp_channels,
            final_inp_channels,
        ]

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, num_channels_pre_layer[i])[
                                1
                            ],
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[
                                1
                            ],
                            nn.ReLU(),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, in_channels)[1],
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self, stages_spec, stage_index, in_channels, multiscale_output=True
    ):
        num_modules = stages_spec["num_modules"][stage_index]
        num_branches = stages_spec["num_branches"][stage_index]
        num_blocks = stages_spec["num_blocks"][stage_index]
        reduce_ratio = stages_spec["reduce_ratios"][stage_index]
        with_fuse = stages_spec["with_fuse"][stage_index]
        module_type = stages_spec["module_type"][stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                LiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp,
                )
            )
            in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if hasattr(m, "norm3"):
                        constant_init(m.norm3, 0)
                    elif hasattr(m, "norm2"):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        """Forward function."""
        x = self.stem(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, "transition{}".format(i))
            for j in range(self.stages_spec["num_branches"][i]):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, "stage{}".format(i))(x_list)

        x = y_list
        if self.with_head:
            x = self.head_layer(x)

        height, width = x[3].size(2), x[3].size(3)
        x0 = F.interpolate(
            x[0], size=(height, width), mode="bilinear", align_corners=True
        )
        x1 = F.interpolate(
            x[1], size=(height, width), mode="bilinear", align_corners=True
        )
        x2 = F.interpolate(
            x[2], size=(height, width), mode="bilinear", align_corners=True
        )
        x = torch.cat([x0, x1, x2, x[3]], dim=1)
        x_dict = {"out4": x}
        return x_dict

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def litehrnet(**kwargs):
    import yaml
    from easydict import EasyDict

    with open("models/backbones/config_litehrnet18.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    model = LiteHRNet(config.extra, **kwargs)
    if "pretrained" in config and config["pretrained"] is not None:
        print("Loading pretrained weights from {}".format(config["pretrained"]))
        pretrained = torch.load(config["pretrained"])
        state_dict = pretrained["state_dict"]
        # get all keys that start with 'backbone.'
        state_dict = {
            k[len("backbone.") :]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        if not config["extra"]["with_head"]:
            # remove head weights
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("head_layer.")
            }
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    import torch

    extra = dict(
        stem=dict(stem_channels=16, out_channels=32, expand_ratio=1),
        num_stages=3,
        stages_spec=dict(
            num_modules=[1, 1, 3],
            num_branches=[2, 3, 4],
            num_blocks=[2, 2, 2],
            num_channels=[(32, 64), (32, 64, 128), (32, 64, 128, 256)],
            reduce_ratios=[8, 8, 8],
            with_fuse=[True, True, True],
            module_type=["LITE", "LITE", "LITE"],
        ),
        with_head=False,
    )

    model = LiteHRNet(extra)
    model.init_weights()
    model.eval()
    inputs = torch.rand(1, 3, 256, 256)
    level_outputs = model(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))
