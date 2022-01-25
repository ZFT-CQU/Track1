import torch.nn as nn
import torchvision.models as models
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from collections import OrderedDict
from typing import Dict


def ResNet_FPN(backbone_name,
               pretrained,
               norm_layer=misc_nn_ops.FrozenBatchNorm2d,
               trainable_layers=3,
               returned_layers=None,
               extra_blocks=None):

    backbone = ResNet(backbone_name,
                      pretrained=pretrained,
                      norm_layer=norm_layer)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1',
                       'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {
        f'layer{k}': str(v)
        for v, k in enumerate(returned_layers)
    }

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2 * 2**(i - 1) for i in returned_layers
    ]
    out_channels = 256
    return BackboneWithFPN(backbone,
                           return_layers,
                           in_channels_list,
                           out_channels,
                           extra_blocks=extra_blocks)


def ResNet(name, pretrained=False, norm_layer=nn.BatchNorm2d):
    if name == 'resnet18':
        return models.resnet18(pretrained=pretrained, norm_layer=norm_layer)
    if name == 'resnet34':
        return models.resnet34(pretrained=pretrained, norm_layer=norm_layer)
    if name == 'resnet50':
        return models.resnet50(pretrained=pretrained, norm_layer=norm_layer)
    if name == 'resnet101':
        return models.resnet101(pretrained=pretrained, norm_layer=norm_layer)
    if name == 'resnet152':
        return models.resnet152(pretrained=pretrained, norm_layer=norm_layer)
    if name == 'resnext50_32x4d':
        return models.resnext50_32x4d(pretrained=pretrained,
                                      norm_layer=norm_layer)
    if name == 'resnext101_32x8d':
        return models.resnext101_32x8d(pretrained=pretrained,
                                       norm_layer=norm_layer)
    if name == 'wide_resnet50_2':
        return models.wide_resnet50_2(pretrained=pretrained,
                                      norm_layer=norm_layer)
    if name == 'wide_resnet101_2':
        return models.wide_resnet101_2(pretrained=pretrained,
                                       norm_layer=norm_layer)


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self,
                 backbone,
                 return_layers,
                 in_channels_list,
                 out_channels,
                 extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone,
                                            return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str,
                                                             str]) -> None:
        if not set(return_layers).issubset(
                [name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
