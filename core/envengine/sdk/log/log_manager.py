import logging
import asyncio
import sys
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque

import colorlog


class LogManager:
    """通用日志管理器"""

    def __init__(self, max_logs: int = 10000, color_enabled: bool = True):
        self.logs = deque(maxlen=max_logs)  # 日志存储
        self.log_handlers: List[callable] = []  # 日志处理器
        self.color_enabled = color_enabled  # 是否启用颜色

        # 清除所有现有的 handlers（避免重复）
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()

        self._setup_logging()

    def _setup_logging(self):
        """配置日志系统"""
        # 创建控制台处理器（默认）
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)

        # 创建 stderr 处理器（ERROR 及以上级别）
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        # 定义日志输出格式和颜色
        if self.color_enabled:
            # 使用带颜色的格式
            console_format = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            # 使用不带颜色的普通格式
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        stdout_handler.setFormatter(console_format)
        stderr_handler.setFormatter(console_format)

        # 创建自定义处理器
        self._internal_handler = _InternalLogHandler(self)
        self._internal_handler.setLevel(logging.INFO)

        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)
        root_logger.addHandler(self._internal_handler)

        logging.info("日志管理器初始化完成")

    def add_handler(self, handler: callable):
        """添加日志处理器"""
        self.log_handlers.append(handler)

    def remove_handler(self, handler: callable):
        """移除日志处理器"""
        if handler in self.log_handlers:
            self.log_handlers.remove(handler)

    def on_log(self, handler: callable):
        """装饰器：注册日志处理器"""
        self.log_handlers.append(handler)
        return handler

    def process_log(self, log_entry: Dict[str, Any]):
        """处理日志条目"""
        # 存储日志
        self.logs.append(log_entry)

        # 调用所有处理器
        for handler in self.log_handlers:
            try:
                handler(log_entry)
            except Exception as e:
                logging.error(f"日志处理器错误: {e}")

    def get_logs(self, count: Optional[int] = None) -> List[Dict]:
        """获取日志"""
        if count:
            return list(self.logs)[-count:]
        return list(self.logs)

    def clear_logs(self):
        """清空日志"""
        self.logs.clear()

    def filter_logs(self, level: Optional[str] = None,
                    module: Optional[str] = None,
                    name: Optional[str] = None) -> List[Dict]:
        """过滤日志"""
        result = list(self.logs)
        if level:
            result = [log for log in result if log['level'] == level]
        if module:
            result = [log for log in result if log['module'] == module]
        if name:
            result = [log for log in result if log['name'] == name]
        return result


class _InternalLogHandler(logging.Handler):
    """内部日志处理器"""

    def __init__(self, log_manager: LogManager):
        super().__init__()
        self.log_manager = log_manager

    def emit(self, record: logging.LogRecord):
        """处理日志"""
        log_entry = {
            "time": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "thread": record.threadName,
            "process": record.processName
        }

        self._process_log(log_entry)

    def _process_log(self, log_entry: Dict):
        """异步处理日志"""
        self.log_manager.process_log(log_entry)
