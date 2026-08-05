from typing import (
    Dict,
    List,
    Type,
    TypeVar,
    Union
)

_T = TypeVar("_T")

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class JSONProtocol:
    def to_json(self) -> str:
        ...

    @classmethod
    def convert_json(cls: Self, instances: Union[Self, List[Self], Dict[str, Self]]) -> str:
        ...

    @classmethod
    def from_json(cls: Self, string: str, is_map: bool = False) -> Union[Self, List[Self], Dict[str, Self]]:
        ...

    def to_dict(self) -> Dict:
        ...

    @classmethod
    def from_dict(cls: Self, d: Dict) -> Self:
        ...


try:
    from dataclasses_json import dataclass_json


    def auto_json(cls_: Type[_T] = None) -> Type[_T]:
        def wrapper(i_cls_: Type[_T]) -> Type[_T]:
            i_cls_ = dataclass_json(i_cls_)

            def group_dumps(cls: Type[_T], instances: Union[_T, List[_T], Dict[str, _T]]) -> str:
                if isinstance(instances, list):
                    return cls.schema().dumps(instances, many=True)
                if isinstance(instances, dict):
                    from json import dumps
                    return dumps({key: val.to_dict() for key, val in instances.items()})
                return instances.to_json()

            i_cls_.convert_json = classmethod(group_dumps)

            deserializer = i_cls_.from_json.__func__  # type: ignore

            def new_deserializer(
                    cls: Type[_T], content: str, is_map: bool = False
            ) -> Union[_T, List[_T], Dict[str, _T]]:
                if is_map:
                    from json import loads
                    datas = loads(content)
                    return {key: i_cls_.from_dict(value) for key, value in datas.items()}
                if content.startswith("["):
                    return cls.schema().loads(content, many=True)
                return deserializer(i_cls_, content)

            i_cls_.from_json = classmethod(new_deserializer)
            return i_cls_

        if cls_ is None:
            return wrapper
        return wrapper(cls_)

except ImportError:
    from abc import ABC
    from dataclasses import fields
    from typing import (
        Any,
        get_args,
        get_origin,
    )
    from json import loads, dumps


    def to_json(instance) -> str:
        if not hasattr(instance, "to_dict"):
            raise TypeError(f"{instance.__class__.__name__} is not JSON serializable.")
        return dumps(instance.to_dict())


    def convert_json(_: Type[_T], instances: Union[_T, List[_T], Dict[str, _T]]) -> str:
        if isinstance(instances, list):
            return dumps([instance.to_dict() for instance in instances])
        if isinstance(instances, dict):
            return dumps({key: val.to_dict() for key, val in instances.items()})
        return instances.to_json()


    def from_json(cls: Type[_T], content: str, is_map: bool = False) -> Union[_T, List[_T], Dict[str, _T]]:
        if not hasattr(cls, "from_dict"):
            raise TypeError(f"{cls.__name__} is not JSON serializable.")
        datas = loads(content)
        if is_map:
            return {key: from_dict(cls, data) for key, data in datas.items()}
        if isinstance(datas, list):
            return [from_dict(cls, data) for data in datas]
        return cls.from_dict(datas)


    def serialize_value(value: object) -> object:
        if isinstance(value, (bool, int, float, bytes, str)):
            return value
        if isinstance(value, list):
            return [serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: serialize_value(value) for key, value in value.items()}
        if getattr(value, "to_dict", None) is not None:
            return getattr(value, "to_dict")()
        return value


    def deserialize_value(type_: Union[type, Type], key: str, value: Any) -> object:
        if value is None:
            if get_origin(type_) is Union and type(None) in get_args(type_):
                return None
            else:
                raise TypeError(f"Key `{key}` is required, got None.")
        if isinstance(type_, TypeVar):
            if len(type_.__constraints__) != 0:
                type_ = type_.__constraints__
            elif type_.__bound__ is not None:
                type_ = type_.__bound__
            else:
                type_ = Any
        if type_ in (bool, int, float, bytes, str):
            return type_(value)
        if get_origin(type_) is list:
            if not isinstance(value, list):
                raise TypeError(f"Key `{key}` expected a list, got {type(value)}")
            return [deserialize_value(get_args(type_)[0], f"{key}[{index}]", item) for index, item in enumerate(value)]
        if get_origin(type_) is dict:
            if not isinstance(value, dict):
                raise TypeError(f"Key `{key}` expected a dict, got {type(value)}")
            key_type, value_type = get_args(type_)
            return {
                deserialize_value(
                    key_type, f"{key}.{index}(key)", index
                ): deserialize_value(
                    value_type, f"{key}.{index}", item
                ) for index, item in value.items()
            }
        this_from_dict = getattr(type_, "from_dict", None)
        if this_from_dict is not None:
            if not isinstance(value, dict):
                raise TypeError(f"Key `{key}` expected a dict, got {type(value)}")
            return this_from_dict(value)
        return value


    def to_dict(instance) -> Dict[str, Any]:
        dict_ = {}
        for field in fields(instance):
            dict_[field.name] = serialize_value(getattr(instance, field.name))
        return dict_


    def from_dict(cls: Type[_T], dictionary: Dict[str, Any]) -> _T:
        dict_ = {}
        for field in fields(cls):
            if field.name not in dictionary:
                continue
            dict_[field.name] = deserialize_value(field.type, field.name, dictionary[field.name])
        return cls(**dict_)


    def auto_json(cls_: Type[_T] = None) -> Type[_T]:
        def wrapper(i_cls_: Type[_T]) -> Type[_T]:
            i_cls_.to_json = to_json
            i_cls_.convert_json = classmethod(convert_json)
            i_cls_.from_json = classmethod(from_json)
            i_cls_.to_dict = to_dict
            i_cls_.from_dict = classmethod(from_dict)

            return i_cls_

        if cls_ is None:
            return wrapper
        return wrapper(cls_)
