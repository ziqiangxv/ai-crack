from ai_crack.net.vnet import VNet
import torch

t = torch.rand(1, 1, 64, 64, 64)
model = VNet()
result = model(t)
print(result.shape)
