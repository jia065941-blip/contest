# -*-coding:utf-8 -*-

class Simulator:
    """
    仿真器装饰器，用于构建字符串与仿真器类的映射关系
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(simulator_class):
            cls._registry[name] = simulator_class
            return simulator_class

        return decorator

    @classmethod
    def create(cls, name, *args, **kwargs):
        simulator_class = cls._registry.get(name)
        if simulator_class:
            return simulator_class(*args, **kwargs)
        raise ValueError(f"未知类型: {name}")
