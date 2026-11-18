"""Physical snapshots and organizer previews from real initialized entities."""

from contextlib import ExitStack
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pygame

from envengine.environment.training_env import TrainingEnv
from envengine.render.renderer import Renderer
from previewer import Previewer


ROOT = Path(__file__).resolve().parents[1]
E01 = ROOT / "scenarios/E01.json"


def vector(x=0, y=0, z=0):
    return SimpleNamespace(x=x, y=y, z=z)


def entity(identifier, kind, *, children=(), visible=False, health=0):
    return SimpleNamespace(
        id=identifier, entityType=kind, nameChn="物理实体", sideId=1,
        lla=vector(1, 2, 3000), posEcf=vector(3, 4, 5),
        velEcf=vector(6, 7, 8), velNue=vector(9, 10, 11),
        survivePoints=health, isVisible=visible, stage=-1,
        parentId=-1, childrenId=list(children),
        detectInfo={}, commRangeInfo={},
    )


class EnvironmentObservationTests(unittest.TestCase):
    def test_live_fields_and_nested_containers_are_copied_without_judging_data(self):
        parent = entity(1, 9500, children=(2, 3))
        interceptor, radar, red = entity(2, 24000), entity(3, 44000), entity(4, 21000)
        interceptor.parentId = radar.parentId = 1
        red.detectInfo = {1: {"entity_id": 1, "pos": {"x": 1, "y": 2, "z": 3000}}}
        red.commRangeInfo = {4: {"pos": {"x": 3, "y": 4, "z": 5}}}
        area = SimpleNamespace(xMin=0, xMax=3, yMin=0, yMax=3)
        polygons = [[[0, 0], [1, 0], [1, 1]]]
        profile = SimpleNamespace(imagineProfile=SimpleNamespace(
            mapArea=area, redArea=SimpleNamespace(coordinates=polygons), simTime=0, simStep=1,
        ))
        factory = SimpleNamespace(get_all_simulators=lambda: [
            SimpleNamespace(entity_ext=SimpleNamespace(entity=value))
            for value in (parent, interceptor, radar, red)
        ])
        env = object.__new__(TrainingEnv)
        env.current_step = 0
        env.engine = SimpleNamespace(
            profile=profile, simulator_factory=factory, sim_time=42,
        )
        env.config = SimpleNamespace(deployment=SimpleNamespace(
            coordinates=polygons, launch_altitude=3000,
        ))
        state = env._get_observation()
        self.assertEqual(state["sim_time_ms"], 42)
        self.assertEqual(state["entities"][1]["inventory"], {"interceptors": 1})
        self.assertEqual(state["entities"][3]["detection_range_m"], 20000)
        self.assertEqual(state["entities"][4]["communication_range_m"], 50000)
        self.assertFalse(state["entities"][1]["isVisible"])
        self.assertNotIn("velocity_nue", state["entities"][1])
        self.assertEqual(state["entities"][1]["id"], 1)
        self.assertEqual(state["entities"][1]["nameChn"], "物理实体")
        self.assertEqual(state["entities"][4]["detectInfo"], red.detectInfo)
        self.assertEqual(state["entities"][4]["commRangeInfo"], red.commRangeInfo)
        parent.posEcf.x = 999
        parent.velEcf.x = 999
        parent.childrenId.clear()
        red.detectInfo[1]["pos"]["x"] = 999
        red.commRangeInfo[4]["pos"]["x"] = 999
        polygons[0][0][0] = 999
        area.xMin = 999
        self.assertEqual(state["entities"][1]["pos"]["x"], 3)
        self.assertEqual(state["entities"][1]["vel"]["x"], 6)
        self.assertEqual(state["entities"][1]["children_ids"], [2, 3])
        self.assertEqual(state["deployment_area"]["coordinates"][0][0][0], 0)
        self.assertEqual(state["map_area"]["xMin"], 0)
        self.assertEqual(state["entities"][4]["detectInfo"][1]["pos"]["x"], 1)
        self.assertEqual(state["entities"][4]["commRangeInfo"][4]["pos"]["x"], 3)
        json.dumps(state, allow_nan=False)
        forbidden = {"profile", "external", "score", "objective_weights", "case_info", "reward_policy", "rules"}

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            else:
                self.assertIsInstance(value, (int, float, str, bool, type(None)))
        check(state)


class OffscreenRenderTests(unittest.TestCase):
    def test_physical_entities_keep_margins_while_logical_nodes_do_not_affect_bounds(self):
        state = {
            "map_area": {"xMin": 1, "xMax": 2, "yMin": 1, "yMax": 2},
            "deployment_area": {"coordinates": []},
            "entities": {
                1: {"id": 1, "type": 9500, "side": 1, "nameChn": "暗巢", "isVisible": False,
                    "health": 0, "pos": {"x": 9, "y": 8}},
                2: {"id": 2, "type": 9202, "side": 0, "isVisible": False,
                    "health": 0, "pos": {"x": 1.1, "y": 1.1}},
                3: {"id": 3, "type": 35000, "side": 1, "nameChn": "虫族控制节点", "isVisible": True,
                    "health": 100, "pos": {"x": -3, "y": -4}},
            },
        }
        original = deepcopy(state)
        font_state, display_state = pygame.font.get_init(), pygame.display.get_init()
        with ExitStack() as stack:
            for target in ("pygame.init", "pygame.display.init", "pygame.display.set_mode",
                           "envengine.render.renderer.Renderer.start"):
                stack.enter_context(patch(target, side_effect=AssertionError("Preview must stay off screen")))
            draw = stack.enter_context(patch.object(Renderer, "_draw_entities", autospec=True,
                                                    side_effect=Renderer._draw_entities))
            hive = stack.enter_context(patch.object(Renderer, "draw_dark_hive", autospec=True,
                                                    side_effect=Renderer.draw_dark_hive))
            eye = stack.enter_context(patch.object(Renderer, "draw_eye", autospec=True,
                                                   side_effect=Renderer.draw_eye))
            png = Renderer.render(state)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (1180, 800))
        self.assertEqual(state, original)
        self.assertEqual(pygame.font.get_init(), font_state)
        self.assertEqual(pygame.display.get_init(), display_state)
        self.assertTrue(draw.call_args.kwargs["show_all"])
        self.assertEqual(set(draw.call_args.args[2]), {1, 2})
        renderer = draw.call_args.args[0]
        for data in state["entities"].values():
            if data["type"] == 35000:
                continue
            x, y = renderer._pos_to_screen(data["pos"]["x"], data["pos"]["y"])
            self.assertTrue(75 <= x <= 925, (x, y))
            self.assertTrue(75 <= y <= 725, (x, y))
        logical_x, logical_y = renderer._pos_to_screen(-3, -4)
        self.assertFalse(0 <= logical_x <= 1000 and 0 <= logical_y <= 800)
        self.assertEqual(hive.call_count, 2)  # One hidden/dead map unit plus its legend entry.
        self.assertEqual(eye.call_count, 2)
        self.assertEqual(eye.call_args_list[0].args[2:4], renderer._pos_to_screen(1.1, 1.1))
        logical = dict(state["entities"][3], pos={
            "x": renderer.center_x, "y": renderer.center_y,
        })
        with patch.object(renderer, "_draw_side_background") as background:
            renderer._draw_entities(pygame.Surface((1000, 800)), {3: logical})
        background.assert_not_called()

    def test_font_initialization_is_restored_on_drawing_failure(self):
        state = {"map_area": {"xMin": 0, "xMax": 1, "yMin": 0, "yMax": 1},
                 "deployment_area": {"coordinates": []}, "entities": {}}
        before = pygame.font.get_init()
        with patch.object(Renderer, "_draw_entities", side_effect=RuntimeError("drawing failed")):
            with self.assertRaisesRegex(RuntimeError, "drawing failed"):
                Renderer.render(state)
        self.assertEqual(pygame.font.get_init(), before)

    def test_preview_preserves_chinese_names_and_uses_the_runtime_legend(self):
        state = {"map_area": {"xMin": 0, "xMax": 1, "yMin": 0, "yMax": 1},
                 "deployment_area": {"coordinates": []},
                 "entities": {7: {"id": 7, "type": 44000, "nameChn": "视野塔",
                                   "pos": {"x": 0.5, "y": 0.5}},
                              8: {"id": 8, "type": 35000, "nameChn": "虫族控制节点",
                                  "pos": {"x": -3, "y": -4}}}}

        with patch.object(Renderer, "_draw_entities") as draw, \
             patch.object(Renderer, "draw_legend2") as legend:
            Renderer.render(state)
        self.assertEqual(draw.call_args.args[1][7]["nameChn"], "视野塔")
        self.assertNotIn(8, draw.call_args.args[1])
        self.assertEqual(state["entities"][7]["nameChn"], "视野塔")
        self.assertEqual(state["entities"][8]["nameChn"], "虫族控制节点")
        self.assertEqual(legend.call_args.kwargs, {})


class PreviewerTests(unittest.TestCase):
    def test_e01_preview_repeats_without_policies_steps_deployment_or_file_writer(self):
        with ExitStack() as stack:
            forbidden = (
                "envengine.engine.engine.Engine.step",
                "envengine.agent_manager.agent_manager.AgentManager.get_deployment_observation",
                "envengine.agent_manager.agent_manager.AgentManager.apply_deployment",
                "blue_defense.shared.controller.BlueDefenseController.__init__",
                "blue_defense.blue_checker.check_blue_policy",
                "scenarios.loader._reward_policy_from_scene",
                "user_agents.red_checker.check_red_agent",
                "user_agents.red_checker.check_deployment",
                "envengine.agent_manager.agent_manager.AgentManager.register_user_agent",
                "envengine.environment.training_env.write_config",
                "envengine.environment.training_env.write_state",
                "envengine.environment.training_env.write_ai_action",
                "envengine.simulator.simulator_factory.write_event",
                "envengine.sdk.writer.async_file_writer.init_writer",
            )
            for target in forbidden:
                stack.enter_context(patch(target, side_effect=AssertionError(f"Unexpected preview side effect: {target}")))
            stack.enter_context(patch("builtins.print"))
            first_png, first = Previewer(E01).preview()
            second_png, second = Previewer(E01).preview()
        self.assertEqual(first, second)
        self.assertEqual(first_png, second_png)
        self.assertEqual(first["step"], 0)
        initial_time = json.loads(E01.read_text(encoding="utf-8"))["imagineProfile"]["simTime"]
        self.assertEqual(first["sim_time_ms"], initial_time)
        self.assertEqual(len(first["entities"]), 234)
        self.assertEqual(sum(value["type"] == 9500 for value in first["entities"].values()), 10)
        self.assertEqual(sum(not value["isVisible"] and value["side"] == 1
                             for value in first["entities"].values()), 85)
        self.assertEqual(sum(value["inventory"]["interceptors"] for value in first["entities"].values()), 78)
        self.assertTrue(all(value["detectInfo"] == {} and value["commRangeInfo"] == {}
                            for value in first["entities"].values()))
        self.assertEqual(first_png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", first_png[16:24]), (1180, 800))
        self.assertEqual(pygame.image.load(BytesIO(first_png)).get_size(), (1180, 800))
        json.dumps(first, ensure_ascii=False, allow_nan=False)
        commander = next(value for value in first["entities"].values() if value["type"] == 35000)
        self.assertLess(commander["pos"]["y"], first["map_area"]["yMin"])

    def test_reset_state_is_rendered_directly_and_errors_close_the_environment(self):
        for stage in ("success", "reset", "render"):
            with self.subTest(stage=stage):
                env = Mock()
                state = {"step": 0, "sim_time_ms": 42, "entities": {},
                         "map_area": {"xMin": 0, "xMax": 1, "yMin": 0, "yMax": 1},
                         "deployment_area": {"coordinates": []}}
                env.reset.return_value = state
                error = RuntimeError(f"{stage} failed")
                scene = object()
                with patch("previewer.load_scenario", return_value=scene) as load, \
                     patch("previewer.TrainingEnv", return_value=env) as create, \
                     patch("envengine.render.renderer.Renderer.render", return_value=b"png") as render:
                    if stage == "success":
                        png, returned = Previewer(E01).preview()
                        self.assertEqual(png, b"png")
                        self.assertIs(returned, state)
                    else:
                        {"reset": env.reset, "render": render}[stage].side_effect = error
                        with self.assertRaises(RuntimeError) as raised:
                            Previewer(E01).preview()
                        self.assertIs(raised.exception, error)
                if stage == "reset":
                    render.assert_not_called()
                else:
                    render.assert_called_once_with(state)
                    self.assertIs(render.call_args.args[0], state)
                env.close.assert_called_once_with()
                load.assert_called_once_with(E01.resolve(), include_rules=False)
                create.assert_called_once_with(scene, record_logs=False)


if __name__ == "__main__":
    unittest.main()
