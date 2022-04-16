from torch import nn
from torch.hub import load_state_dict_from_url
import torch
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SE(nn.Module):
    def __init__(self, planes, ratio=16):
        super(SE, self).__init__()
        mip = max(8, planes // ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=mip, kernel_size=1, bias=False),
            nn.BatchNorm2d(mip),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mip, out_channels=planes, kernel_size=1),
        )
        
    def forward(self, x):
        out = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        out = self.fusion(out)
        return x * out.sigmoid()


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, kernel_size//2, groups=groups, bias=False),
            norm_layer(out_planes),
            # nn.ReLU(inplace=True)
            # HSwish()
            Swish()
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, att=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if isinstance(att, str):
            if "se" in att:
                att_block = SE(hidden_dim, int(att.split("_")[1]))
        else:
            att_block = nn.Sequential()

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # att
            att_block,
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 att=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv_first = ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)

        # building inverted residual blocks
        features = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer, att=att))
                input_channel = output_channel
        self.features = nn.Sequential(*features)

        self.conv_last = ConvBNReLU(input_channel, last_channel, kernel_size=1, norm_layer=norm_layer)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_channel, 1280, bias=False)
        self.bn = nn.BatchNorm1d(1280)
        # self.act = nn.ReLU()
        # self.act = HSwish()
        self.act = Swish()
        self.drop = nn.Dropout(0.2)
        self.classifier = nn.Linear(1280, num_classes, bias=False)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)


    def _forward_impl(self, x):
        x = self.conv_first(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.gap(x).flatten(1)
        x = self.drop(self.act(self.bn(self.fc(x))))
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

