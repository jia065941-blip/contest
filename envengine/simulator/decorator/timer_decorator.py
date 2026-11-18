# -*-coding:utf-8 -*-


import functools
import time


def timer_decorator(func):
    """计时装饰器"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 更高精度
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"[{func.__name__}] 执行时间: {elapsed:.10f} 秒")
        return result

    return wrapper
