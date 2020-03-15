import torch as t
from vgg import VGG
vgg = VGG(21)
input_data = t.randn(1, 3, 224, 224)
scores = vgg(input_data)
print(scores)
