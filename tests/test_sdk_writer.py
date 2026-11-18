import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from envengine.sdk.writer import (
    WriteConfig, get_writer, init_writer, write_ai_action, write_config,
    write_event, write_immediately, write_state,
)
from envengine.sdk.writer import async_file_writer as writer_module


class SdkWriterTests(unittest.TestCase):
    def setUp(self):
        previous = writer_module.AsyncJsonFileWriter._instance
        if previous is not None:
            previous.close()
        self.temporary = tempfile.TemporaryDirectory(prefix="competition-writer-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def writer(self, name="run", **overrides):
        options = {"output_dir": str(self.directory / name), "verbose": False}
        options.update(overrides)
        writer = init_writer(WriteConfig(**options))
        self.addCleanup(writer.close)
        return writer

    def records(self, filename, name="run"):
        return [json.loads(line) for line in (self.directory / name / filename).read_text(encoding="utf-8").splitlines()]

    def test_explicit_initialization_after_close_creates_independent_session(self):
        first = self.writer("first")
        write_event({"round": 1}, "events.json")
        first.close()
        first.close()
        self.assertIs(get_writer(), first)
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            write_event({"round": 99}, "unexpected.json")
        write_immediately()
        self.assertFalse((self.directory / "first" / "unexpected.json").exists())

        second = self.writer("second")
        self.assertIsNot(second, first)
        first.close()
        write_event({"round": 2}, "events.json")
        second.close()
        self.assertEqual(self.records("events.json", "first"), [{"round": 1, "data_type": "event"}])
        self.assertEqual(self.records("events.json", "second"), [{"round": 2, "data_type": "event"}])

    def test_disabled_channels_create_no_directory_and_leave_inputs_unchanged(self):
        writer = self.writer(
            enable_config=False, enable_state=False, enable_event=False,
            enable_ai_action=False, log_frequency=10,
        )
        data = {"nested": {"values": [1]}}
        expected = copy.deepcopy(data)
        for write in (write_config, write_state, write_event, write_ai_action):
            write(data, "disabled.json")
        self.assertEqual(data, expected)
        self.assertEqual(writer.buffer, [])
        writer.flush()
        writer.close()
        self.assertFalse((self.directory / "run").exists())

    def test_explicit_initialization_flushes_implicit_session_and_applies_disabled_config(self):
        writer_module.AsyncJsonFileWriter._instance = None
        implicit_config = WriteConfig(output_dir=str(self.directory / "implicit"), verbose=False)
        with patch.object(writer_module, "WriteConfig", return_value=implicit_config):
            implicit = get_writer()
        self.addCleanup(implicit.close)
        implicit.add({"session": "implicit"}, "state", "state.json")

        explicit = self.writer(
            "explicit", enable_config=False, enable_state=False,
            enable_event=False, enable_ai_action=False,
        )
        self.assertIsNot(explicit, implicit)
        self.assertTrue(implicit._closed)
        self.assertTrue(implicit.executor._shutdown)
        self.assertEqual(self.records("state.json", "implicit"), [{"session": "implicit"}])
        self.assertEqual(explicit.output_dir, self.directory / "explicit")
        for write in (write_config, write_state, write_event, write_ai_action):
            write({"session": "explicit"}, "disabled.json")
        explicit.close()
        self.assertFalse((self.directory / "explicit").exists())

    def test_each_enabled_helper_preserves_input_and_captures_nested_snapshot(self):
        writer = self.writer()
        for index, write in enumerate((write_config, write_state, write_event, write_ai_action)):
            with self.subTest(write=write.__name__):
                data = {"step": 2, "data_type": "caller", "nested": {"values": [index]}}
                expected = copy.deepcopy(data)
                write(data, str(index) + ".json")
                self.assertEqual(data, expected)
                data["nested"]["values"].append(99)
        writer.flush()
        for index, data_type in enumerate(("config", "state", "event", "aiAction")):
            self.assertEqual(self.records(str(index) + ".json"), [{
                "step": 2, "data_type": data_type, "nested": {"values": [index]},
            }])

    def test_direct_add_captures_snapshot_and_skips_disabled_copy(self):
        writer = self.writer(enable_event=False)

        class CannotCopy:
            def __deepcopy__(self, memo):
                raise AssertionError("disabled data was copied")

        self.assertFalse(writer.add({"value": CannotCopy()}, "event", "disabled.json"))
        data = {"values": [1]}
        self.assertFalse(writer.add(data, "state", "state.json"))
        data["values"].append(2)
        writer.flush()
        self.assertEqual(self.records("state.json"), [{"values": [1]}])

    def test_state_sampling_and_channel_switches_keep_only_requested_records(self):
        writer = self.writer(log_frequency=3, enable_event=False, enable_ai_action=False)
        for step in range(7):
            write_state({"step": step}, "state.json")
        write_config({"version": 1}, "config.json")
        write_event({"ignored": True}, "events.json")
        write_ai_action({"ignored": True}, "actions.json")
        write_ai_action(None, "actions.json")
        writer.flush()
        self.assertEqual([record["step"] for record in self.records("state.json")], [0, 3, 6])
        self.assertEqual(self.records("config.json"), [{"version": 1, "data_type": "config"}])
        self.assertFalse((self.directory / "run" / "events.json").exists())
        self.assertFalse((self.directory / "run" / "actions.json").exists())

    def test_buffers_keep_files_separate_across_automatic_and_explicit_flushes(self):
        writer = self.writer(batch_size=3)
        self.assertFalse(writer.add({"n": 1}, "state", "a.json"))
        self.assertFalse(writer.add({"n": 2}, "state", "nested/b.json"))
        self.assertTrue(writer.add({"n": 3}, "state", "a.json"))
        self.assertFalse(writer.add({"n": 4}, "state", "nested/b.json"))
        writer.flush()
        self.assertEqual(self.records("a.json"), [{"n": 1}, {"n": 3}])
        self.assertEqual(self.records("nested/b.json"), [{"n": 2}, {"n": 4}])
        self.assertEqual(writer.buffer, [])

    def test_flush_reports_asynchronous_batch_failure(self):
        writer = self.writer(batch_size=1)
        with patch.object(writer, "_write_batch", side_effect=OSError("disk failure")):
            writer.add({"n": 1}, "state", "state.json")
            with self.assertRaisesRegex(OSError, "disk failure"):
                writer.flush()
        writer.close()

    def test_close_reports_failure_and_releases_executor_before_next_session(self):
        writer = self.writer()
        writer.add({"n": 1}, "state", "state.json")
        with patch.object(writer, "_write_batch", side_effect=OSError("close failure")):
            with self.assertRaisesRegex(OSError, "close failure"):
                writer.close()
        self.assertTrue(writer._closed)
        self.assertTrue(writer.executor._shutdown)
        writer.close()
        next_writer = self.writer("next")
        write_event({"n": 2}, "event.json")
        next_writer.close()
        self.assertEqual(self.records("event.json", "next"), [{"n": 2, "data_type": "event"}])

    def test_close_cleans_up_when_batch_submission_itself_fails(self):
        writer = self.writer()
        writer.add({"n": 1}, "state", "state.json")
        with patch.object(writer.executor, "submit", side_effect=RuntimeError("submit failure")):
            with self.assertRaisesRegex(RuntimeError, "submit failure"):
                writer.close()
        self.assertTrue(writer._closed)
        self.assertTrue(writer.executor._shutdown)
        writer.close()

    def test_real_filesystem_failure_is_visible_on_flush(self):
        obstruction = self.directory / "blocked"
        obstruction.write_text("file", encoding="utf-8")
        writer = self.writer(output_dir=str(obstruction / "child"))
        writer.add({"n": 1}, "state", "state.json")
        with self.assertRaises(OSError):
            writer.flush()
        writer.close()

    def test_constructor_failure_starts_no_executor_or_active_session(self):
        config = WriteConfig(output_dir=None, verbose=False)
        with patch.object(writer_module, "ThreadPoolExecutor") as executor:
            with self.assertRaises(TypeError):
                init_writer(config)
        executor.assert_not_called()
        self.assertTrue(writer_module.AsyncJsonFileWriter._instance._closed)
        self.writer("valid").close()

    def test_quiet_writer_emits_no_info_messages(self):
        with patch.object(writer_module.logging, "info") as info:
            writer = self.writer()
            write_event({"n": 1}, "event.json")
            writer.flush()
            writer.close()
        info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
