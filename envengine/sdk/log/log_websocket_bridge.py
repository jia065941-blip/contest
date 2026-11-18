import asyncio
import json
from typing import Optional

from envengine.sdk.Transport import WebSocketServer
from envengine.sdk.log import LogManager


class LogWebSocketBridge:
    """日志和WebSocket的结合器"""

    def __init__(self, ws_server: WebSocketServer, log_manager: LogManager):
        self.ws_server = ws_server
        self.log_manager = log_manager
        self.enabled = False

    def start(self):
        """开始转发日志到WebSocket"""
        self.enabled = True
        # 注册日志处理器
        self.log_manager.on_log(self._forward_log)

        print("日志WebSocket桥接已启动")

    def stop(self):
        """停止转发"""
        self.enabled = False
        self.log_manager.remove_handler(self._forward_log)
        print("日志WebSocket桥接已停止")

    def _forward_log(self, log_entry: dict):
        """转发日志到WebSocket"""
        if not self.enabled:
            return

        # 格式化为WebSocket消息
        message = {
            "type": "log",
            "data": log_entry,
            "timestamp": log_entry.get("time")
        }

        # 广播给所有WebSocket客户端
        self.ws_server.broadcast(message)

    async def _handle_ws_message(self, websocket, message: str):
        """处理WebSocket客户端发来的消息"""
        try:
            data = json.loads(message)
            cmd = data.get("cmd")

            if cmd == "get_logs":
                # 客户端请求历史日志
                count = data.get("count", 100)
                logs = self.log_manager.get_logs(count)
                await self.ws_server.send_to_client(websocket, {
                    "type": "logs_history",
                    "data": logs
                })
            elif cmd == "clear_logs":
                # 客户端请求清空日志
                self.log_manager.clear_logs()
                await self.ws_server.broadcast({
                    "type": "system",
                    "message": "日志已清空"
                })
            elif cmd == "filter_logs":
                # 客户端请求过滤日志
                filters = data.get("filters", {})
                logs = self.log_manager.filter_logs(**filters)
                await self.ws_server.send_to_client(websocket, {
                    "type": "filtered_logs",
                    "data": logs
                })

        except json.JSONDecodeError:
            await self.ws_server.send_to_client(websocket, {
                "type": "error",
                "message": "无效的JSON格式"
            })