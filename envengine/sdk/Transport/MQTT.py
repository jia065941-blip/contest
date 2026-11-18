# -*-coding:utf-8 -*-
from typing import Callable, Tuple
import paho.mqtt.client as mqtt


class MQTT(object):
    def __init__(self, name: str, mqtt_ip: str, mqtt_port: int, on_subscribe: Callable[[str, str], None]):
        r"""
        Mqtt初始化
        :param name: Mqtt节点名称
        :param mqtt_ip: MqttIp
        :param on_subscribe: 接收到消息回调, function(str, str) -> NoReturn
        """
        self.client = mqtt.Client(client_id=name, transport='tcp')
        self.client.will_set("willTopic", payload=-1, qos=0, retain=False)
        self.client.connect(mqtt_ip, mqtt_port, 60)
        self.on_subscribe = on_subscribe
        self.client.on_connect = lambda client, userdata, flags, rc: print("Mqtt connected return: {0}".format(rc))
        self.client.on_message = lambda client, userdata, msg: self.on_subscribe(msg.topic, msg.payload.decode("utf-8"))
        self.client.loop_start()

    def publish(self, topic: str, msg: str, qos: int) -> None:
        r"""
        消息发送
        :param qos:
        :param topic: 主题
        :param msg: 负载
        :return: NoReturn
        """
        payload = msg
        try:
            self.client.publish(topic, payload, qos=qos)
        except Exception as e:
            print(e)

    def subscribe(self, topic: str, qos: int) -> Tuple[int, int]:
        r"""
        主题订阅
        :param qos:
        :param topic: 主题
        :return: RESULT
        """
        res = self.client.subscribe(topic, qos=qos)
        return res

    def unsubscribe(self, topic: str) -> Tuple[int, int]:
        r"""
        取消订阅
        :param topic: 主题
        :return: RESULT
        """
        res = self.client.unsubscribe(topic)
        return res
