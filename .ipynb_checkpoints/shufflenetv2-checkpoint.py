import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class HardSwish(nn.Module):
    def forward(self, x):
        return x * torch.clamp((x + 1) / 2, min=0, max=1)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class ShiftSigmoid(nn.Module):
    def forward(self, x):
        return 2 * torch.sigmoid(x) - 1


class Act(nn.Module):
    def __init__(self, out_planes=None, act_type="relu"):
        super(Act, self).__init__()

        self.act = None
        if act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            self.act = nn.PReLU(out_planes)
        elif act_type == "swish":
            self.act = Swish()
        elif act_type == "hardswish":
            self.act = HardSwish()
        elif act_type == "mish":
            self.act = Mish()

    def forward(self, x):
        return self.act(x)


class Scale(nn.Module):
    def __init__(self, inplanes, outplanes, reduce=0, scale=None):
        super(Scale, self).__init__()
        if reduce == 0:
            self.context = nn.AdaptiveAvgPool2d(1)
            self.fusion1 = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        elif reduce > 0:
            self.context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=inplanes, out_channels=inplanes//reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(inplanes//reduce),
                nn.ReLU(inplace=True)
            )
            self.fusion1 = nn.Sequential(
                nn.Conv2d(in_channels=inplanes//reduce, out_channels=outplanes, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        self.scale = None
        if scale == "sigmoid":
            self.scale = nn.Sigmoid()
        elif scale == "shiftsigmoid":
            self.scale = ShiftSigmoid()
        elif scale == "tanh":
            self.scale = nn.Tanh()
        elif scale == "softmax":
            self.scale = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.context(x)
        scale = self.fusion1(out)
        if self.scale is not None:
            scale = self.scale(scale)
        return x * scale


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, ksize, stride, act_type):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            Act(mid_channels, act_type),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            Scale(mid_channels, mid_channels, reduce=0, scale="sigmoid"),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            Act(outputs, act_type),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                Act(inp, act_type)
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
            
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=1000, model_size='1.5x', dropout=0.2, act_type="relu"):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 52, 104, 208, 640]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            Act(input_channel, act_type)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel, mid_channels=output_channel // 2, ksize=3, stride=2, act_type=act_type))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel, mid_channels=output_channel // 2, ksize=3, stride=1, act_type=act_type))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.last_conv = nn.Sequential(
            nn.Conv2d(self.stage_out_channels[-2], 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            Act(1280, act_type)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, 1280)
        self.bn = nn.BatchNorm1d(1280)
        self.act = Act(1280, act_type)
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.last_conv(x).pow(2)

        x = self.gap(x).flatten(1)
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = ShuffleNetV2()
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())
