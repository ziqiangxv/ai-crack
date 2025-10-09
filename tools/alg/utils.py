import numpy as np
import time
from functools import wraps

from ai_crack.utils import color_print

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        color_print(f"--- {func.__name__}: {execution_time:.3f} seconds", 'blue')
        return result

    return wrapper

class TimerContext:
    def __init__(self, description: str = "Timer"):
        """
        初始化定时器。

        :param description: 定时器的描述信息，默认为 "Timer"
        """
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """
        进入上下文时调用，记录开始时间。
        """
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出上下文时调用，记录结束时间并打印耗时。
        """
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        color_print(f"--- {self.description}: {elapsed_time:.3f} seconds", 'blue')
