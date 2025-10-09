import torch
import torch.nn.functional as F

# 设置更全面的打印选项
torch.set_printoptions(
    linewidth=150,     # 每行最大字符数
    precision=3,       # 浮点数精度
    threshold=10000,   # 显示元素总数阈值（超过则截断）
    edgeitems=3,       # 每维开头和结尾显示的元素数
    sci_mode=False     # 禁用科学计数法
)

def resize_with_pad(slice: torch.Tensor, target_size) -> torch.Tensor:
    """保持纵横比的填充调整"""

    if slice.shape[-2:] == target_size:
        return slice

    # slice = slice.unsqueeze(0)

    h, w = slice.shape[-2:]
    target_h, target_w = target_size

    # 计算缩放比例
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 使用双线性插值调整大小
    resized = F.interpolate(
                slice,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )

    print(resized)

    # 计算填充尺寸
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    # 应用填充 (填充顺序: 左, 右, 上, 下)
    padded = F.pad(
                resized,
                [pad_left, pad_right, pad_top, pad_bottom],
                mode='constant',
                value=0
            )

    # 移除批次和通道维度 [H, W]
    return padded

t = torch.randn(1, 1, 3, 4)
target_size = (6, 6)

print(t)
print(t.shape)
t_resize = resize_with_pad(t, target_size)


print(t_resize.shape)
print(t_resize)
