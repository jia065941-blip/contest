"""SDK 日志管理器的级别、回调与根日志作用域。"""

from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import logging
from pathlib import Path
import unittest
from unittest.mock import patch


# 日志模块可独立验证，不必载入原生仿真模型。
SOURCE = Path(__file__).resolve().parents[1] / "envengine/sdk/log/log_manager.py"
SPEC = spec_from_file_location("_sdk_logging_under_test", SOURCE)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
LogManager = MODULE.LogManager


class TrackingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []
        self.close_calls = 0

    def emit(self, record):
        self.messages.append(record.getMessage())

    def close(self):
        self.close_calls += 1
        super().close()


class LogManagerTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.original_handlers = list(self.root.handlers)
        self.original_level = self.root.level
        for handler in self.original_handlers:
            self.root.removeHandler(handler)
        self.previous_handler = TrackingHandler()
        self.root.addHandler(self.previous_handler)
        self.root.setLevel(logging.ERROR)
        self.managers = []
        self.other_handlers = []
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.streams = patch.multiple("sys", stdout=self.stdout, stderr=self.stderr)
        self.streams.start()
        self.addCleanup(self.restore_logging)

    def restore_logging(self):
        try:
            for manager in reversed(self.managers):
                manager.close()
        finally:
            for handler in list(self.root.handlers):
                self.root.removeHandler(handler)
            self.previous_handler.close()
            for handler in self.other_handlers:
                handler.close()
            for handler in self.original_handlers:
                self.root.addHandler(handler)
            self.root.setLevel(self.original_level)
            self.streams.stop()

    def manager(self, **kwargs):
        manager = LogManager(color_enabled=False, **kwargs)
        self.managers.append(manager)
        return manager

    def test_default_level_and_close_restore_previous_root_configuration(self):
        manager = self.manager()
        installed = list(self.root.handlers)
        self.assertEqual(self.root.level, logging.INFO)
        self.assertIn("日志管理器初始化完成", self.stdout.getvalue())
        self.assertEqual(len(installed), 3)
        self.assertNotIn(self.previous_handler, installed)
        self.assertEqual(self.previous_handler.close_calls, 0)

        manager.close()

        self.assertEqual(self.root.handlers, [self.previous_handler])
        self.assertEqual(self.root.level, logging.ERROR)
        self.assertEqual(self.previous_handler.close_calls, 0)
        self.assertTrue(all(handler._closed for handler in installed))
        self.assertEqual(manager.get_logs()[0]["message"], "日志管理器初始化完成")

    def test_warning_level_hides_internal_info_and_splits_warning_error(self):
        manager = self.manager(level=logging.WARNING)
        logging.info("internal info")
        logging.warning("visible warning")
        logging.error("visible error")

        self.assertEqual([entry["message"] for entry in manager.get_logs()],
                         ["visible warning", "visible error"])
        self.assertNotIn("日志管理器初始化完成", self.stdout.getvalue())
        self.assertNotIn("internal info", self.stdout.getvalue())
        self.assertIn("visible warning", self.stdout.getvalue())
        self.assertNotIn("visible error", self.stdout.getvalue())
        self.assertIn("visible error", self.stderr.getvalue())
        self.assertNotIn("visible warning", self.stderr.getvalue())

    def test_named_info_logger_can_show_run_messages_at_warning_root_level(self):
        manager = self.manager(level=logging.WARNING)
        logger = logging.getLogger("sdk.logging.tests.simulator")
        previous_level = logger.level
        self.addCleanup(logger.setLevel, previous_level)
        logger.setLevel(logging.INFO)

        logger.info("run started")

        self.assertIn("run started", self.stdout.getvalue())
        self.assertEqual(manager.get_logs()[0]["message"], "run started")

    def test_close_preserves_external_handlers_added_during_the_scope(self):
        manager = self.manager(level=logging.WARNING)
        later = TrackingHandler()
        self.other_handlers.append(later)
        self.root.addHandler(later)
        logging.warning("during scope")

        manager.close()
        logging.error("after scope")

        self.assertEqual(set(self.root.handlers), {self.previous_handler, later})
        self.assertEqual(self.previous_handler.messages, ["after scope"])
        self.assertEqual(later.messages, ["during scope", "after scope"])
        self.assertEqual(self.previous_handler.close_calls, 0)
        self.assertEqual(later.close_calls, 0)

    def test_close_is_idempotent_and_does_not_touch_later_configuration(self):
        manager = self.manager()
        installed = list(self.root.handlers)
        with patch.object(installed[0], "close", wraps=installed[0].close) as close:
            manager.close()
            self.root.setLevel(logging.CRITICAL)
            later = TrackingHandler()
            self.other_handlers.append(later)
            self.root.addHandler(later)
            manager.close()
            self.assertEqual(close.call_count, 1)

        self.assertEqual(self.root.level, logging.CRITICAL)
        self.assertEqual(set(self.root.handlers), {self.previous_handler, later})

    def test_sequential_runs_do_not_accumulate_handlers(self):
        for level in (logging.INFO, logging.WARNING, logging.INFO):
            with self.subTest(level=level):
                manager = self.manager(level=level)
                self.assertEqual(len(self.root.handlers), 3)
                logging.warning("one run")
                self.assertEqual(len(manager.filter_logs(level="WARNING")), 1)
                manager.close()
                self.assertEqual(self.root.handlers, [self.previous_handler])
                self.assertEqual(self.root.level, logging.ERROR)

    def test_nested_scopes_restore_the_outer_manager_before_the_caller(self):
        outer = self.manager(level=logging.INFO)
        outer_handlers = list(self.root.handlers)
        inner = self.manager(level=logging.WARNING)
        logging.warning("inner")
        inner.close()
        self.assertEqual(self.root.handlers, outer_handlers)
        self.assertEqual(self.root.level, logging.INFO)
        logging.info("outer")
        self.assertEqual([entry["message"] for entry in inner.get_logs()], ["inner"])
        self.assertEqual(outer.get_logs()[-1]["message"], "outer")
        outer.close()
        self.assertEqual(self.root.handlers, [self.previous_handler])

    def test_invalid_level_leaves_the_callers_configuration_intact(self):
        with self.assertRaises(ValueError):
            self.manager(level="not a logging level")
        self.assertEqual(self.root.handlers, [self.previous_handler])
        self.assertEqual(self.root.level, logging.ERROR)
        self.assertEqual(self.previous_handler.close_calls, 0)

    def test_close_releases_all_owned_handlers_even_if_one_close_raises(self):
        manager = self.manager()
        installed = list(self.root.handlers)
        with patch.object(installed[-1], "close", side_effect=RuntimeError("close failed")):
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                manager.close()
        self.assertTrue(all(handler._closed for handler in installed[:-1]))
        self.assertEqual(self.root.handlers, [self.previous_handler])
        self.assertEqual(self.root.level, logging.ERROR)
        manager.close()

    def test_cache_callbacks_and_filtering_remain_available(self):
        manager = self.manager(level=logging.WARNING)
        received = []

        @manager.on_log
        def forward(entry):
            received.append(entry)

        logging.warning("first")
        manager.remove_handler(forward)
        logging.error("second")
        self.assertEqual([entry["message"] for entry in received], ["first"])
        self.assertEqual(manager.filter_logs(level="WARNING"), received)
        self.assertEqual(manager.get_logs(count=1)[0]["message"], "second")
        manager.clear_logs()
        self.assertEqual(manager.get_logs(), [])


if __name__ == "__main__":
    unittest.main()
