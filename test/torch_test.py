import torch
import torch.nn.functional as F


# t = torch.rand((1, 3, 3))
# t.unsqueeze_(0)
# t.unsqueeze_(0)
# print(t.shape)


# a = torch.rand((1, 2, 5, 5))
# print(a)
# print(a[..., 0:3, 0:3])

# import torch
# import torch.nn.functional as F

# # 假设这是 VNet 输出的 feature_map，形状为 (batch_size, num_classes, height, width, depth)
# num_classes = 2
# feature_map = torch.randn(1, num_classes, 1, 3, 3)

# # 选取概率最大的类别作为预测类别
# mask = torch.argmax(feature_map, dim=1)

# print("Feature map shape:", feature_map.shape)
# print("Mask shape:", mask.shape)
# print(feature_map)
# print(mask)


a = torch.rand((1, 2, 1, 2, 3))
print(a.squeeze().shape)
