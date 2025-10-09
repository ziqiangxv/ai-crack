from functools import lru_cache

@lru_cache
def expensive_function(x):
    print(f"计算 {x}...")
    return x * x

# 第一次调用会执行函数
print(expensive_function(5))  # 输出: 计算 5... \n 25

# 第二次用相同参数调用，直接返回缓存结果
print(expensive_function(5))  # 输出: 25 (没有"计算..."输出)