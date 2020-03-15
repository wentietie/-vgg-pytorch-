import torch as t
from inceptionv1 import Inceptionv1
in_net = Inceptionv1(3, 64, 32, 64, 64, 96, 32)
in_data = t.randn(1, 3, 256, 256)
output = in_net(in_data)
print(output)
