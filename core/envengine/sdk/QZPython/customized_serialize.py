__all__ = [
    'auto_binary',
    'auto_json',
    'SerializeProtocol',
    "serialize_test"
]

from envengine.sdk.QZPython.customized_binary import auto_binary, BinaryProtocol
from envengine.sdk.QZPython.customized_json import auto_json, JSONProtocol


class SerializeProtocol(BinaryProtocol, JSONProtocol):
    ...


def __json_test(cls, datas: dict) -> bool:
    if len(datas) == 0:
        return False
    key, value = next(iter(datas.items()))

    value = cls(**value)
    content = value.to_json()
    begin = cls.from_json(content)

    if value != begin:
        print("第一个值的JSON序列化与反序列化结果不一致")
        return False

    object_list = [cls(**value) for _, value in datas.items()]
    content = cls.convert_json(object_list)
    new_list = cls.from_json(content)
    if object_list != new_list:
        print("数组JSON序列化与反序列化结果不一致")
        return False

    object_dict = {key: cls(**value) for key, value in datas.items()}
    content = cls.convert_json(object_dict)
    new_dict = cls.from_json(content, is_map=True)
    if object_dict != new_dict:
        print("字典JSON序列化与反序列化结果不一致")
        return False

    return True

def __binary_test(cls, datas: dict) -> bool:
    if len(datas) == 0:
        return False
    key, value = next(iter(datas.items()))

    value = cls(**value)
    content = value.to_binary()
    begin = cls.from_binary(content)

    if value != begin:
        print("第一个值的Binary序列化与反序列化结果不一致")
        return False

    object_list = [cls(**value) for _, value in datas.items()]
    content = cls.convert_binary(object_list)
    new_list = cls.from_binary(content)
    if object_list != new_list:
        print("数组Binary序列化与反序列化结果不一致")
        return False

    object_dict = {key: cls(**value) for key, value in datas.items()}
    content = cls.convert_binary(object_dict)
    new_dict = cls.from_binary(content)
    if object_dict != new_dict:
        print("字典Binary序列化与反序列化结果不一致")
        return False

    return True


def serialize_test(cls, datas: dict) -> bool:
    ans = True

    if getattr(cls, "from_json", None) is None:
        print("当前类型不支持自动JSON序列化, 跳过JSON验证.")
    else:
        ans = ans and __json_test(cls, datas)

    if getattr(cls, "from_binary", None) is None:
        print("当前类型不支持自动Binary序列化, 跳过Binary验证.")
    else:
        ans = ans and __binary_test(cls, datas)

    if ans:
        print(cls.__name__, "测试成功")
    else:
        print(cls.__name__, "测试失败")

    return ans
