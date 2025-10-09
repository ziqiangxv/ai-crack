# -*- coding:utf-8 -*-

''''''

import sys

if sys.platform == 'win32':
    sys.path.append('../ai_crack')

import torch

# 设置更全面的打印选项
torch.set_printoptions(
    linewidth=150,     # 每行最大字符数
    precision=3,       # 浮点数精度
    threshold=10000,   # 显示元素总数阈值（超过则截断）
    edgeitems=3,       # 每维开头和结尾显示的元素数
    sci_mode=False     # 禁用科学计数法
)

