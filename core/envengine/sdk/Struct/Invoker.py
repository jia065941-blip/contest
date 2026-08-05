# -*-coding:utf-8 -*-

from ..Transport import MQTT
from .LogicInput import LogicInput
from .Package import Package
from .ReceiveSaver import ReceiveSaver
from typing import Callable


class Invoker:
    """调用器"""

    def __init__(self, client: MQTT, stream: bool, is_async: bool):
        self.__client__: MQTT = client
        """MQTT"""
        self.__is_stream__: bool = stream
        """是否为流"""
        self.__is_async__: bool = is_async
        """是否为异步"""
        self.__result__: Package | None = None
        """上次的结果"""
        self.__receive__: ReceiveSaver = ReceiveSaver()
        """收到的值"""
        self.__current_round__: int = 0
        """当前轮次"""

    def save_payload(self, payload: str):
        """保存负载"""
        if self.__is_async__:
            self.__receive__.save(Package(0, payload))
        else:
            self.__receive__.save(Package.deserialize(payload))

    def save_package(self, package: Package):
        """保存包"""
        self.__receive__.save(package)

    def invoke(self, topic: str, logic: Callable[[LogicInput], str]) -> None:
        """调用"""
        content = self.__receive__.load()
        if content is None:
            return
        if self.__result__ is None or content.seq != self.__result__.seq or self.__is_async__:
            try:
                result = logic(LogicInput(self.__current_round__, content.content, self.__is_stream__))
                self.__result__ = Package(content.seq, result)
            except Exception as e:
                print("Logic function call error. Please check your function.\n",
                      "    State: ", content.content + "\n",
                      "    Error: ", str(e))
        if self.__is_async__:
            payload = self.__result__.content
        else:
            payload = self.__result__.serialize()
        self.__client__.publish(topic, payload, 0)

    def reset(self, round_: int):
        self.__result__ = None
        self.__receive__.load()
        self.__current_round__ = round_
