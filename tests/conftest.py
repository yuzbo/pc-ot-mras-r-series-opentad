import sys
import types

import torch
import torch.nn as nn
from mmengine.registry import MODELS as MMENGINE_MODELS


def _identity(*args, **kwargs):
    return None


def _get_sinusoid_encoding(*args, **kwargs):
    return torch.zeros(1, 1, 1)


def _install_mmaction_stubs():
    mmaction = types.ModuleType("mmaction")
    mmaction_models = types.ModuleType("mmaction.models")
    mmaction_backbones = types.ModuleType("mmaction.models.backbones")
    mmaction_registry = types.ModuleType("mmaction.registry")
    mmaction_utils = types.ModuleType("mmaction.utils")
    mmaction_swin = types.ModuleType("mmaction.models.backbones.swin")
    mmaction_vit_mae = types.ModuleType("mmaction.models.backbones.vit_mae")

    mmaction_registry.MODELS = MMENGINE_MODELS
    mmaction_utils.ConfigType = dict
    mmaction_utils.OptConfigType = dict

    dummy_class_names = [
        "PatchEmbed3D",
        "PatchMerging",
        "WindowAttention3D",
        "Mlp",
        "SwinTransformer3D",
        "ResNet3dSlowFast",
    ]
    for name in dummy_class_names:
        setattr(mmaction_swin, name, type(name, (nn.Module,), {"__init__": lambda self, *a, **k: nn.Module.__init__(self)}))

    for name in ["get_window_size", "compute_mask", "window_partition", "window_reverse"]:
        setattr(mmaction_swin, name, _identity)
    mmaction_vit_mae.get_sinusoid_encoding = _get_sinusoid_encoding

    mmaction.models = mmaction_models
    mmaction.registry = mmaction_registry
    mmaction.utils = mmaction_utils
    mmaction_models.backbones = mmaction_backbones
    mmaction_backbones.swin = mmaction_swin
    mmaction_backbones.vit_mae = mmaction_vit_mae

    sys.modules.setdefault("mmaction", mmaction)
    sys.modules.setdefault("mmaction.registry", mmaction_registry)
    sys.modules.setdefault("mmaction.utils", mmaction_utils)
    sys.modules.setdefault("mmaction.models", mmaction_models)
    sys.modules.setdefault("mmaction.models.backbones", mmaction_backbones)
    sys.modules.setdefault("mmaction.models.backbones.swin", mmaction_swin)
    sys.modules.setdefault("mmaction.models.backbones.vit_mae", mmaction_vit_mae)


_install_mmaction_stubs()


def _install_torchvision_stubs():
    torchvision = types.ModuleType("torchvision")
    torchvision_models = types.ModuleType("torchvision.models")
    torchvision_video = types.ModuleType("torchvision.models.video")
    torchvision_resnet = types.ModuleType("torchvision.models.video.resnet")
    torchvision_ops = types.ModuleType("torchvision.ops")

    class _DummyVideoResNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _DummyBlock(nn.Module):
        expansion = 1

        def __init__(self, *args, **kwargs):
            super().__init__()

    def _roi_align(input, boxes, output_size, aligned=True):
        del boxes, aligned
        return torch.zeros((1, input.shape[1], output_size[0], output_size[1]), dtype=input.dtype, device=input.device)

    torchvision_resnet.VideoResNet = _DummyVideoResNet
    torchvision_resnet.R2Plus1dStem = _DummyBlock
    torchvision_resnet.BasicBlock = _DummyBlock
    torchvision_ops.roi_align = _roi_align
    torchvision.models = torchvision_models
    torchvision.ops = torchvision_ops
    torchvision_models.video = torchvision_video
    torchvision_video.resnet = torchvision_resnet

    sys.modules.setdefault("torchvision", torchvision)
    sys.modules.setdefault("torchvision.models", torchvision_models)
    sys.modules.setdefault("torchvision.models.video", torchvision_video)
    sys.modules.setdefault("torchvision.models.video.resnet", torchvision_resnet)
    sys.modules.setdefault("torchvision.ops", torchvision_ops)


_install_torchvision_stubs()


class _NMS1DCPU:
    @staticmethod
    def nms(segs, scores, iou_threshold=0.0):
        return scores.argsort(descending=True)

    @staticmethod
    def softnms(segs, scores, dets, iou_threshold=0.0, sigma=0.5, min_score=0.0, method=2, t1=0.0, t2=0.0):
        order = scores.argsort(descending=True)
        dets[: order.numel(), :2] = segs[order]
        dets[: order.numel(), 2] = scores[order]
        return order


sys.modules.setdefault("nms_1d_cpu", _NMS1DCPU)


class _Align1DStub:
    @staticmethod
    def forward(input, roi, feature_dim, ratio):
        del ratio
        return input.new_zeros((roi.shape[0], input.shape[1], feature_dim))

    @staticmethod
    def backward(grad_output, rois, feature_dim, bs, ch, t, ratio):
        del rois, feature_dim, ratio
        return grad_output.new_zeros((bs, ch, t))


class _BoundaryMaxPoolingStub:
    @staticmethod
    def forward(input, segments):
        return input.new_zeros((segments.shape[0], input.shape[1], 2))

    @staticmethod
    def backward(grad_output, input, segments):
        del grad_output, segments
        return input.new_zeros(input.shape)


sys.modules.setdefault("Align1D", _Align1DStub)
sys.modules.setdefault("boundary_max_pooling_cuda", _BoundaryMaxPoolingStub)
