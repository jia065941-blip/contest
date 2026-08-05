import asyncio
import logging
import time

from envengine.sdk.Transport import WebSocketServer
from envengine.sdk.log import LogManager, LogWebSocketBridge

ws_server = WebSocketServer(host="0.0.0.0", port=8765)
log_manager = LogManager()

# 创建桥接器
bridge = LogWebSocketBridge(ws_server, log_manager)


async def main():
    # 启动WebSocket服务器
    await ws_server.start()

    # 启动日志桥接（开始推送日志到WebSocket）
    bridge.start()

    # 正常使用日志
    logger = logging.getLogger(__name__)
    logger.info("系统启动")

    # 模拟运行
    for i in range(1000000):
        await asyncio.sleep(1)
        logger.warning(f"这是第{i + 1}条日志")

    # 停止桥接（可选）
    # bridge.stop()

    # 停止服务器
    await ws_server.stop()


if __name__ == '__main__':
    asyncio.run(main())
