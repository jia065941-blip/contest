# -*-coding:utf-8 -*-
import logging
from typing import Any

from envengine.sdk.base_struct.Message import Command

logger = logging.getLogger(__name__)


class CommandAdapter:
    """
    指令适配器
    """

    @staticmethod
    def common_adapter(ai_commands: list) -> list[Command]:
        """
        通用适配
        :param commands:
        :return:
        """
        command_convert_list = []
        for command in ai_commands:
            if not command:
                continue
            if "executor_id" not in command:
                logger.warning(f"[CommandAdapter] 无效的指令：{command}, 原因：缺失executor_id, 系统将忽略该指令！")
                continue
            if "commandType_id" not in command:
                logger.warning(f"[CommandAdapter] 无效的指令：{command}, 原因：缺失commandType_id, 系统将忽略该指令！")
                continue
            if not CommandAdapter.get_command_attributes(command):
                logger.warning(f"[CommandAdapter] 无效的指令参数：{command}, 原因：缺失除executor_id和commandType_id以外的参数, 系统将忽略该指令！")
                continue
            command_convert = Command(
                executorId=command["executor_id"],
                commandTypeId=command["commandType_id"],
                commandAttributes=CommandAdapter.get_command_attributes(command)
            )
            command_convert_list.append(command_convert)
        return command_convert_list

    @staticmethod
    def get_command_attributes(command: dict) -> dict:
        """
        获取指令参数
        :param command:
        :return:
        """
        filtered_command = {k: v for k, v in command.items()
                            if k not in ['executor_id', 'commandType_id']}
        return filtered_command
