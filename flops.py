import torch
from thop import profile, clever_format
from mobilenetv2 import MobileNetV2
from shufflenetv2 import ShuffleNetV2
from mobilenext import MobileNeXt

input = torch.randn(1, 3, 224, 224)

# train_model = MobileNetV2(width_mult=1., att=None)
# train_model = ShuffleNetV2(att="se_4")
train_model = MobileNeXt(width_mult=1., att="se_4")
print(train_model)
train_model.eval()      # Don't forget to call this before inference.


macs, params = profile(train_model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")

print('Flops:  ', macs)
print('Params: ', params)
