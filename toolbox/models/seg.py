import torch
import torch.nn as nn

from .backbone_vit import EfficientViTBackbone
from .nn import (
    ConvLayer,
    DAGBlock,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpSampleLayer,
)
from .utils import build_kwargs_from_config

__all__ = [
    "EfficientViTSeg",
    "efficientvit_seg_b0",
    "efficientvit_seg_b1",
    "efficientvit_seg_b2",
    "efficientvit_seg_b3",
]


class SegHead(DAGBlock):
    def __init__(
        self,
        fid_list: list,
        in_channel_list: list,
        stride_list: list,
        head_stride: int,
        head_width: int,
        head_depth: int,
        expand_ratio: float,
        final_expand: float or None,
        n_classes: int,
        dropout=0,
        norm="bn2d",
        act_func="hswish",
    ):
        inputs = {}
        for fid, in_channel, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None),
                        UpSampleLayer(factor=factor),
                    ]
                )

        middle = []
        for _ in range(head_depth):
            block = MBConv(
                head_width,
                head_width,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=(act_func, act_func, None),
            )
            middle.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle)

        outputs = {
            "segout": OpSequential(
                [
                    None
                    if final_expand is None
                    else ConvLayer(head_width, head_width * final_expand, 1, norm=norm, act_func=act_func),
                    ConvLayer(
                        head_width * (final_expand or 1),
                        n_classes,
                        1,
                        use_bias=True,
                        dropout=dropout,
                        norm=None,
                        act_func=None,
                    ),
                ]
            )
        }

        super(SegHead, self).__init__(inputs, "add", None, middle=middle, outputs=outputs)


class EfficientViTSeg(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone, head: SegHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.head(feed_dict)

        return feed_dict["segout"]


def efficientvit_seg_b0(dataset: str, **kwargs) -> EfficientViTSeg:
    from .backbone_vit import efficientvit_backbone_b0

    backbone = efficientvit_backbone_b0(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[128, 64, 32],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=32,
            head_depth=1,
            expand_ratio=4,
            final_expand=4,
            n_classes=9,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b1(dataset: str, **kwargs) -> EfficientViTSeg:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1

    backbone = efficientvit_backbone_b1(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[256, 128, 64],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=3,
            expand_ratio=4,
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[256, 128, 64],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=64,
            head_depth=3,
            expand_ratio=4,
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b2(dataset: str, **kwargs) -> EfficientViTSeg:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2

    backbone = efficientvit_backbone_b2(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[384, 192, 96],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=96,
            head_depth=3,
            expand_ratio=4,
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[384, 192, 96],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=96,
            head_depth=3,
            expand_ratio=4,
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model


def efficientvit_seg_b3(dataset: str, **kwargs) -> EfficientViTSeg:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b3

    backbone = efficientvit_backbone_b3(**kwargs)

    if dataset == "cityscapes":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            final_expand=4,
            n_classes=19,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    elif dataset == "ade20k":
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            final_expand=None,
            n_classes=150,
            **build_kwargs_from_config(kwargs, SegHead),
        )
    else:
        raise NotImplementedError
    model = EfficientViTSeg(backbone, head)
    return model