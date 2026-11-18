import sys
from abc import ABC
from dataclasses import fields, is_dataclass
from enum import Enum
from struct import pack, unpack
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    Set,
    Tuple,
    get_args,
    get_origin,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

K = TypeVar('K')
T = TypeVar('T')


class TypeId(Enum):
    BOOL = 0
    INT8_T = 1
    INT16_T = 2
    INT32_T = 3
    INT64_T = 4
    UINT8_T = 5
    UINT16_T = 6
    UINT32_T = 7
    UINT64_T = 8
    FLOAT = 9
    DOUBLE = 10
    LONG_DOUBLE = 11
    CHAR = 12
    STRING = 13
    PTR = 14
    ARRAY = 15
    VECTOR = 16
    LIST = 17
    DEQUE = 18
    MAP = 19
    UNORDERED_MAP = 20
    SET = 21
    UNORDERED_SET = 22
    OPTIONAL = 23
    NAMED_STRUCT = 24


class FastBinaryWriter:
    __slots__ = ('_buffer', '_pos')
    
    def __init__(self, capacity: int = 4096):
        self._buffer = bytearray(capacity)
        self._pos = 0
    
    def _ensure_capacity(self, additional: int):
        required = self._pos + additional
        if required > len(self._buffer):
            new_size = max(len(self._buffer) * 2, required)
            self._buffer.extend(b'\x00' * (new_size - len(self._buffer)))
    
    def write_byte(self, value: int):
        self._ensure_capacity(1)
        self._buffer[self._pos] = value & 0xFF
        self._pos += 1
    
    def write_bytes(self, data: bytes):
        length = len(data)
        self._ensure_capacity(length)
        self._buffer[self._pos:self._pos + length] = data
        self._pos += length
    
    def write_type_id(self, type_id: TypeId):
        self.write_byte(type_id.value)
    
    def write_bool(self, value: bool):
        self.write_byte(1 if value else 0)
    
    def write_int64(self, value: int, signed: bool = True):
        self._ensure_capacity(8)
        value_bytes = value.to_bytes(8, "little", signed=signed)
        self._buffer[self._pos:self._pos + 8] = value_bytes
        self._pos += 8
    
    def write_double(self, value: float):
        self._ensure_capacity(8)
        pack("<d", value) 
        self._buffer[self._pos:self._pos + 8] = pack("<d", value)
        self._pos += 8
    
    def write_string_content(self, value: str):
        encoded = value.encode()
        self.write_type_id(TypeId.INT64_T if len(encoded) >= 0 else TypeId.UINT64_T)
        self.write_int64(len(encoded), signed=(len(encoded) < 0))
        self.write_bytes(encoded)
    
    def to_bytes(self) -> bytes:
        return bytes(self._buffer[:self._pos])


class FastBinaryReader:
    __slots__ = ('_data', '_pos', '_length')
    
    def __init__(self, data: Union[bytes, memoryview]):
        if isinstance(data, memoryview):
            self._data = data
        else:
            self._data = memoryview(data)
        self._pos = 0
        self._length = len(self._data)
    
    def read_byte(self) -> int:
        if self._pos >= self._length:
            raise IndexError("Buffer overflow")
        value = self._data[self._pos]
        self._pos += 1
        return value
    
    def read_bytes(self, size: int) -> bytes:
        if self._pos + size > self._length:
            raise IndexError("Buffer overflow")
        value = bytes(self._data[self._pos:self._pos + size])
        self._pos += size
        return value
    
    def read_type_id(self) -> TypeId:
        return TypeId(self.read_byte())
    
    def read_bool(self) -> bool:
        return bool(self.read_byte())
    
    def read_int(self, size: int, signed: bool) -> int:
        if self._pos + size > self._length:
            raise IndexError("Buffer overflow")
        value = int.from_bytes(self._data[self._pos:self._pos + size], "little", signed=signed)
        self._pos += size
        return value
    
    def read_float(self, size: int) -> float:
        if size == 4:
            value = unpack("<f", self._data[self._pos:self._pos + 4])[0]
            self._pos += 4
        elif size == 8:
            value = unpack("<d", self._data[self._pos:self._pos + 8])[0]
            self._pos += 8
        else:
            raise ValueError(f"Unsupported float size: {size}")
        return value
    
    def read_string_content(self) -> str:
        type_id = self.read_type_id()
        if type_id not in (TypeId.INT8_T, TypeId.INT16_T, TypeId.INT32_T, TypeId.INT64_T,
                          TypeId.UINT8_T, TypeId.UINT16_T, TypeId.UINT32_T, TypeId.UINT64_T):
            raise TypeError(f"Expected int type for string length, got {type_id}")
        
        self._pos -= 1
        size = self.read_int_with_type()
        
        if self._pos + size > self._length:
            raise IndexError("Buffer overflow")
        
        value = bytes(self._data[self._pos:self._pos + size]).decode()
        self._pos += size
        return value
    
    def read_int_with_type(self) -> int:
        type_id = self.read_type_id()
        if type_id == TypeId.INT8_T:
            return self.read_int(1, signed=True)
        elif type_id == TypeId.INT16_T:
            return self.read_int(2, signed=True)
        elif type_id == TypeId.INT32_T:
            return self.read_int(4, signed=True)
        elif type_id == TypeId.INT64_T:
            return self.read_int(8, signed=True)
        elif type_id == TypeId.UINT8_T:
            return self.read_int(1, signed=False)
        elif type_id == TypeId.UINT16_T:
            return self.read_int(2, signed=False)
        elif type_id == TypeId.UINT32_T:
            return self.read_int(4, signed=False)
        elif type_id == TypeId.UINT64_T:
            return self.read_int(8, signed=False)
        raise TypeError(f"Expected INT type, got {type_id}")
    
    @property
    def remaining(self) -> int:
        return self._length - self._pos
    
    @property
    def position(self) -> int:
        return self._pos
    
    @position.setter
    def position(self, value: int):
        self._pos = value


class AutoBinary:
    _field_cache: Dict[Type, tuple] = {}
    
    @staticmethod
    def serialize(instance) -> bytes:
        writer = FastBinaryWriter()
        AutoBinary._serialize(writer, instance)
        return writer.to_bytes()
    
    @staticmethod
    def deserialize(content_: Union[bytes, memoryview], type_: Type[T]) -> Optional[T]:
        reader = FastBinaryReader(content_)
        return AutoBinary._deserialize_type(reader, type_)
    
    @classmethod
    def _get_cached_fields(cls, type_: Type) -> tuple:
        if type_ not in cls._field_cache:
            cls._field_cache[type_] = tuple(fields(type_))
        return cls._field_cache[type_]
    
    @staticmethod
    def _serialize(writer: FastBinaryWriter, instance) -> None:
        if is_dataclass(instance):
            AutoBinary._serialize_data_class(writer, instance)
        elif isinstance(instance, str):
            AutoBinary._serialize_string(writer, instance)
        elif isinstance(instance, bool):
            AutoBinary._serialize_bool(writer, instance)
        elif isinstance(instance, int):
            AutoBinary._serialize_int(writer, instance)
        elif isinstance(instance, float):
            AutoBinary._serialize_float(writer, instance)
        elif isinstance(instance, list):
            AutoBinary._serialize_list(writer, instance)
        elif isinstance(instance, dict):
            AutoBinary._serialize_dict(writer, instance)
        elif isinstance(instance, set):
            AutoBinary._serialize_set(writer, instance)
        elif instance is None:
            AutoBinary._serialize_optional(writer)
    
    @staticmethod
    def _serialize_bool(writer: FastBinaryWriter, instance: bool) -> None:
        writer.write_type_id(TypeId.BOOL)
        writer.write_bool(instance)
    
    @staticmethod
    def _serialize_int(writer: FastBinaryWriter, instance: int) -> None:
        if instance < 0:
            writer.write_type_id(TypeId.INT64_T)
            writer.write_int64(instance, signed=True)
        else:
            writer.write_type_id(TypeId.UINT64_T)
            writer.write_int64(instance, signed=False)
    
    @staticmethod
    def _serialize_float(writer: FastBinaryWriter, instance: float) -> None:
        writer.write_type_id(TypeId.DOUBLE)
        writer.write_double(instance)
    
    @staticmethod
    def _serialize_string(writer: FastBinaryWriter, instance: str) -> None:
        writer.write_type_id(TypeId.STRING)
        writer.write_string_content(instance)
    
    @staticmethod
    def _serialize_optional(writer: FastBinaryWriter) -> None:
        writer.write_type_id(TypeId.OPTIONAL)
    
    @staticmethod
    def _serialize_list(writer: FastBinaryWriter, instance: List[T]) -> None:
        writer.write_type_id(TypeId.LIST)
        AutoBinary._serialize_int(writer, len(instance))
        for element in instance:
            AutoBinary._serialize(writer, element)
    
    @staticmethod
    def _serialize_dict(writer: FastBinaryWriter, instance: Dict[K, T]) -> None:
        writer.write_type_id(TypeId.MAP)
        AutoBinary._serialize_int(writer, len(instance))
        for key, value in instance.items():  # 修复：原版用enumerate是bug
            AutoBinary._serialize(writer, key)
            AutoBinary._serialize(writer, value)
    
    @staticmethod
    def _serialize_set(writer: FastBinaryWriter, instance: Set[T]) -> None:
        writer.write_type_id(TypeId.SET)
        AutoBinary._serialize_int(writer, len(instance))
        for element in instance:
            AutoBinary._serialize(writer, element)
    
    @staticmethod
    def _serialize_data_class(writer: FastBinaryWriter, instance: Type[T]) -> None:
        writer.write_type_id(TypeId.NAMED_STRUCT)
        
        cached_fields = AutoBinary._get_cached_fields(type(instance))
        AutoBinary._serialize_int(writer, len(cached_fields))
        
        for field_ in cached_fields:
            AutoBinary._serialize_string(writer, field_.name)
            AutoBinary._serialize(writer, getattr(instance, field_.name))
    
    @staticmethod
    def _deserialize_type(reader: FastBinaryReader, type_: Type[T]) -> Optional[T]:
        if reader.remaining <= 0:
            return None
            
        if is_dataclass(type_):
            return AutoBinary._deserialize_data_class(reader, type_)
        elif type_ == str:
            return AutoBinary._deserialize_string(reader)
        elif type_ == bool:
            return AutoBinary._deserialize_bool(reader)
        elif type_ == int:
            return AutoBinary._deserialize_int(reader)
        elif type_ == float:
            return AutoBinary._deserialize_float(reader)
        elif get_origin(type_) is Union and type(None) in get_args(type_):
            return AutoBinary._deserialize_optional(reader, type_)
        elif type_ == list or get_origin(type_) is list:
            return AutoBinary._deserialize_list(reader, type_)
        elif type_ == dict or get_origin(type_) is dict:
            return AutoBinary._deserialize_dict(reader, type_)
        elif type_ == set or get_origin(type_) is set:
            return AutoBinary._deserialize_set(reader, type_)
        else:
            return AutoBinary._deserialize_any(reader)
    
    @staticmethod
    def _deserialize_any(reader: FastBinaryReader) -> Tuple[Any, bool]:
        pos = reader.position
        type_id = reader.read_type_id()
        
        reader.position = pos
        
        if type_id in (TypeId.INT8_T, TypeId.INT16_T, TypeId.INT32_T, TypeId.INT64_T,
                      TypeId.UINT8_T, TypeId.UINT16_T, TypeId.UINT32_T, TypeId.UINT64_T):
            return AutoBinary._deserialize_int(reader)
        
        if type_id in (TypeId.FLOAT, TypeId.DOUBLE, TypeId.LONG_DOUBLE):
            return AutoBinary._deserialize_float(reader)
        
        if type_id in (TypeId.CHAR, TypeId.STRING):
            return AutoBinary._deserialize_string(reader)
        
        if type_id in (TypeId.LIST, TypeId.ARRAY, TypeId.VECTOR, TypeId.SET, 
                      TypeId.UNORDERED_SET, TypeId.DEQUE):
            return AutoBinary._deserialize_list(reader, List[Any])
        
        if type_id in (TypeId.MAP, TypeId.UNORDERED_MAP):
            return AutoBinary._deserialize_dict(reader, Dict[Any, Any])
        
        if type_id in (TypeId.NAMED_STRUCT, TypeId.OPTIONAL):
            return None
        
        return None
    
    @staticmethod
    def _deserialize_bool(reader: FastBinaryReader) -> bool:
        type_id = reader.read_type_id()
        if type_id != TypeId.BOOL:
            raise TypeError(f"Deserialize bool error: Expected BOOL type, got {type_id}.")
        return reader.read_bool()
    
    @staticmethod
    def _deserialize_int(reader: FastBinaryReader) -> int:
        return reader.read_int_with_type()
    
    @staticmethod
    def _deserialize_float(reader: FastBinaryReader) -> float:
        type_id = reader.read_type_id()
        if type_id == TypeId.FLOAT:
            return reader.read_float(4)
        elif type_id in (TypeId.DOUBLE, TypeId.LONG_DOUBLE):
            return reader.read_float(8)
        raise TypeError(f"Deserialize float error: Expected FLOAT/DOUBLE type, got {type_id}.")
    
    @staticmethod
    def _deserialize_string(reader: FastBinaryReader) -> str:
        type_id = reader.read_type_id()
        
        if type_id not in (TypeId.STRING, TypeId.CHAR):
            raise TypeError(f"Deserialize string error: Expected STRING/CHAR type, got {type_id}.")
        
        if type_id == TypeId.STRING:
            return reader.read_string_content()
        else:  # CHAR
            return reader.read_bytes(1).decode()
    
    @staticmethod
    def _deserialize_optional(reader: FastBinaryReader, type_: Type[T]) -> Optional[T]:
        type_id = reader.read_type_id()
        
        if type_id == TypeId.OPTIONAL:
            return None
        
        # 获取Optional内层类型
        element_types = [i for i in get_args(type_) if i != type(None)]
        element_type = element_types[0]
        
        reader.position -= 1
        
        return AutoBinary._deserialize_type(reader, element_type)
    
    @staticmethod
    def _deserialize_list(reader: FastBinaryReader, type_: Type[T]) -> List:
        if type_ == List[Any]:
            element_type = Any
        else:
            args = get_args(type_)
            element_type = args[0] if args else Any
        
        type_id = reader.read_type_id()
        if type_id not in (TypeId.LIST, TypeId.ARRAY, TypeId.DEQUE, 
                          TypeId.VECTOR, TypeId.SET, TypeId.UNORDERED_SET):
            raise TypeError(f"Deserialize list error: Expected LIST like type, got {type_id}.")
        
        size = reader.read_int_with_type()
        
        data = [None] * size
        for i in range(size):
            if element_type == Any:
                data[i] = AutoBinary._deserialize_any(reader)
            else:
                data[i] = AutoBinary._deserialize_type(reader, element_type)
        
        return data
    
    @staticmethod
    def _deserialize_dict(reader: FastBinaryReader, type_: Type[T]) -> Dict:
        if type_ == Dict[Any, Any]:
            key_type = Any
            element_type = Any
        else:
            args = get_args(type_)
            key_type = args[0] if args else Any
            element_type = args[1] if len(args) > 1 else Any
        
        type_id = reader.read_type_id()
        if type_id not in (TypeId.MAP, TypeId.UNORDERED_MAP):
            raise TypeError(f"Deserialize dict error: Expected MAP like type, got {type_id}.")
        
        size = reader.read_int_with_type()
        
        data = {}
        for _ in range(size):
            if key_type == Any:
                key = AutoBinary._deserialize_any(reader)
            else:
                key = AutoBinary._deserialize_type(reader, key_type)
            
            if element_type == Any:
                element = AutoBinary._deserialize_any(reader)
            else:
                element = AutoBinary._deserialize_type(reader, element_type)
            
            data[key] = element
        
        return data
    
    @staticmethod
    def _deserialize_set(reader: FastBinaryReader, type_: Type[T]) -> Set:
        args = get_args(type_)
        element_type = args[0] if args else Any
        
        type_id = reader.read_type_id()
        if type_id not in (TypeId.LIST, TypeId.ARRAY, TypeId.DEQUE, 
                          TypeId.VECTOR, TypeId.SET, TypeId.UNORDERED_SET):
            raise TypeError(f"Deserialize set error: Expected SET like type, got {type_id}.")
        
        size = reader.read_int_with_type()
        
        data = set()
        for _ in range(size):
            if element_type == Any:
                element = AutoBinary._deserialize_any(reader)
            else:
                element = AutoBinary._deserialize_type(reader, element_type)
            data.add(element)
        
        return data
    
    @staticmethod
    def _deserialize_data_class(reader: FastBinaryReader, type_: Type[T]) -> Optional[T]:
        type_id = reader.read_type_id()
        if type_id != TypeId.NAMED_STRUCT:
            raise TypeError(f"Deserialize {type_} error: Expected NAMED_STRUCT type.")
        
        size = reader.read_int_with_type()
        
        instance = object.__new__(type_)
        if hasattr(type_, '__init__'):
            try:
                type_.__init__(instance)
            except:
                pass
        
        cached_fields = AutoBinary._get_cached_fields(type_)
        field_dict = {f.name: f for f in cached_fields}
        
        for _ in range(size):
            if reader.remaining <= 0:
                break
                
            name = reader.read_string_content()
            if name not in field_dict:
                raise TypeError(f"Deserialize {type_} error: Field '{name}' not found.")
            
            field_ = field_dict[name]
            data = AutoBinary._deserialize_type(reader, field_.type)
            setattr(instance, name, data)
        
        return instance


class AutoBinaryMixin(ABC):
    def to_binary(self) -> bytes:
        return AutoBinary.serialize(self)

    @classmethod
    def convert_binary(cls, instances: Union[Self, List[Self], Dict[Any, Self]]) -> bytes:
        return AutoBinary.serialize(instances)

    @classmethod
    def from_binary(cls, data: bytes) -> Self:
        instance = AutoBinary.deserialize(data, cls)
        if instance is None:
            raise TypeError(f"Deserialize {cls.__name__} error.")
        return instance

    @classmethod
    def list_from_binary(cls, data: bytes) -> List[Self]:
        instance = AutoBinary.deserialize(data, List[cls])
        if instance is None:
            raise TypeError(f"Deserialize List[{cls.__name__}] error.")
        return instance

    @classmethod
    def dict_from_binary(cls, data: bytes) -> Dict[Any, Self]:
        instance = AutoBinary.deserialize(data, Dict[Any, cls])
        if instance is None:
            raise TypeError(f"Deserialize Dict[Any, {cls.__name__}] error.")
        return instance


def auto_binary(name: str):
    _ = name

    def decorator(cls_: Type[T]) -> Type[T]:
        cls_.to_binary = AutoBinaryMixin.to_binary
        cls_.convert_binary = classmethod(AutoBinaryMixin.convert_binary.__func__)
        cls_.from_binary = classmethod(AutoBinaryMixin.from_binary.__func__)
        cls_.list_from_binary = classmethod(AutoBinaryMixin.list_from_binary.__func__)
        cls_.dict_from_binary = classmethod(AutoBinaryMixin.dict_from_binary.__func__)
        return cls_

    return decorator