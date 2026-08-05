import asyncio
import json
import websockets

from typing import Set, Callable, Any, Optional
import logging

from websockets.legacy.server import WebSocketServerProtocol
logger = logging.getLogger(__name__)


class WebSocketServer:
    """通用 WebSocket 服务端"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.server: Optional[websockets.Server] = None
        self.clients: Set[WebSocketServerProtocol] = set()
        self.message_handlers: list[Callable] = []  # 消息处理器
        self.connection_handlers: list[Callable] = []  # 连接处理器

    async def start(self):
        """启动服务器"""
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        logger.info(f"WebSocket服务器已启动: ws://{self.host}:{self.port}")
        return self.server

    async def stop(self):
        """停止服务器"""
        if self.server:
            # 关闭所有客户端连接
            for client in self.clients:
                await client.close()
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket服务器已停止")

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """处理客户端连接"""
        client_addr = websocket.remote_address
        self.clients.add(websocket)
        logger.info(f"客户端连接: {client_addr}")

        # 触发连接事件
        for handler in self.connection_handlers:
            try:
                await handler(websocket, True)
            except Exception as e:
                logger.error(f"连接处理器错误: {e}")

        try:
            async for message in websocket:
                # 触发消息事件
                for handler in self.message_handlers:
                    try:
                        await handler(websocket, message)
                    except Exception as e:
                        logger.error(f"消息处理器错误: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开: {client_addr}")
        finally:
            self.clients.remove(websocket)
            # 触发断开事件
            for handler in self.connection_handlers:
                try:
                    await handler(websocket, False)
                except Exception as e:
                    logger.error(f"连接处理器错误: {e}")

    def broadcast(self, data: Any):
        """向所有客户端广播消息"""
        if not self.clients:
            return

        # 自动序列化
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)

        asyncio.gather(
            *[client.send(data) for client in self.clients],
            return_exceptions=True
        )

    async def send_to_client(self, websocket: WebSocketServerProtocol, data: Any):
        """向指定客户端发送消息"""
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False)
        await websocket.send(data)

    def on_message(self, handler: Callable):
        """注册消息处理器"""
        self.message_handlers.append(handler)
        return handler

    def on_connection(self, handler: Callable):
        """注册连接处理器"""
        self.connection_handlers.append(handler)
        return handler

    @property
    def client_count(self) -> int:
        """获取客户端数量"""
        return len(self.clients)