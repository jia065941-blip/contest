# -*-coding:utf-8 -*-
import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import main as platform_main
import numpy as np
import pygame
from pyproj import Transformer

from envengine import TrainingEnv
from envengine.simulator.simlulator_impl.CompCruiseMissileHSimulator import (
    CompCruiseMissileHSimulator,
)
from envengine.simulator.simlulator_impl.CompCruiseMissileLSimulator import (
    CompCruiseMissileLSimulator,
)
from envengine.simulator.simlulator_impl.CompCruiseMissileMSimulator import (
    CompCruiseMissileMSimulator,
)
from envengine.simulator.simlulator_impl.InterceptorSimulator import InterceptorSimulator
from .experiment import (
    OBS_LEN,
    ConformalCalibrator,
    LSTMPredictor,
)

RED_SIMULATORS = (
    CompCruiseMissileHSimulator,
    CompCruiseMissileMSimulator,
    CompCruiseMissileLSimulator,
)


def run_main_with_prediction(
    side: str,
    model_dir: str,
    cp_path: str,
    output_dir: str,
) -> None:
    output_path = Path(output_dir) / f"{datetime.now():%Y%m%d%H%M%S}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = OnlinePredictor(model_dir, cp_path, side)
    overlay = PredictionOverlay()
    active_renderer = {"value": None}
    original_step = TrainingEnv.step
    original_flip = pygame.display.flip

    with output_path.open("x", encoding="utf-8") as prediction_file:
        def flip_with_prediction():
            renderer = active_renderer["value"]
            screen = pygame.display.get_surface()
            if renderer is not None and screen is not None:
                overlay.draw(screen, renderer)
            original_flip()

        def step_with_prediction(env, *step_args, **step_kwargs):
            result = original_step(env, *step_args, **step_kwargs)
            predictions = predictor.update(env)
            attribute = (
                "interceptor_trajectory_predictions"
                if side == "blue"
                else "red_trajectory_predictions"
            )
            setattr(env, attribute, predictions)
            if env.renderer is not None:
                active_renderer["value"] = env.renderer
                overlay.update(predictions)
            prediction_file.write(
                json.dumps(
                    {
                        "round": env.current_round,
                        "step": env.current_step,
                        "sim_time_ms": env.engine.sim_time,
                        "side": side,
                        "predictions": predictions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            return result

        TrainingEnv.step = step_with_prediction
        pygame.display.flip = flip_with_prediction
        logging.info(f"[在线轨迹预测] 输出文件: {output_path}")
        try:
            platform_main.main()
        finally:
            TrainingEnv.step = original_step
            pygame.display.flip = original_flip


class OnlinePredictor:
    """缓存指定阵营实体位置，批量预测模型定义的未来轨迹。"""

    def __init__(self, model_dir: str, cp_path: str, side: str = "blue"):
        if side not in ("blue", "red"):
            raise ValueError("side必须为blue或red")
        self.model = LSTMPredictor.load(model_dir)
        self.calibrator = ConformalCalibrator.load(cp_path)
        self.side = side
        self.histories: dict[int, deque[np.ndarray]] = {}
        self.round_id = None

    def update(self, env) -> list[dict]:
        if self.round_id != env.current_round:
            self.histories.clear()
            self.round_id = env.current_round

        active_ids, ready = set(), []
        entity_types = (24000,) if self.side == "blue" else (21000, 21001, 21002)
        for entity_type in entity_types:
            simulators = env.engine.simulator_factory.get_simulators_by_type(entity_type)
            for simulator in simulators:
                item = self._update_simulator(simulator, active_ids)
                if item is not None:
                    ready.append(item)

        for entity_id in self.histories.keys() - active_ids:
            del self.histories[entity_id]
        if not ready:
            return []

        relative = self.model.predict(np.stack([item[2] for item in ready]))
        if relative.shape[1] != len(self.calibrator.radii):
            raise ValueError("模型预测步数与CP半径数量不一致")
        return [
            {
                **item[0],
                "predicted_position_ecf": (prediction + item[1]).tolist(),
                "radius_m": self.calibrator.radii.tolist(),
            }
            for item, prediction in zip(ready, relative)
        ]

    def _update_simulator(self, simulator, active_ids: set[int]):
        entity = simulator.entity_ext.entity
        if self.side == "blue":
            if not isinstance(simulator, InterceptorSimulator):
                raise TypeError(f"实体 {entity.id} 不是拦截弹仿真器")
            if not _active_blue(simulator):
                return None
            metadata = {"interceptor_id": int(entity.id)}
        else:
            if not isinstance(simulator, RED_SIMULATORS):
                raise TypeError(f"实体 {entity.id} 不是红方进攻弹仿真器")
            if not _active_red(simulator):
                return None
            metadata = {
                "missile_id": int(entity.id),
                "entity_type": int(entity.entityType),
            }

        entity_id = int(entity.id)
        active_ids.add(entity_id)
        history = self.histories.setdefault(entity_id, deque(maxlen=OBS_LEN))
        history.append(
            np.asarray(
                [entity.posEcf.x, entity.posEcf.y, entity.posEcf.z],
                dtype=np.float32,
            )
        )
        if len(history) < OBS_LEN:
            return None
        reference = history[-1]
        return metadata, reference, np.asarray(history, dtype=np.float32) - reference


class PredictionOverlay:
    """黄色中心线表示预测，橙色圆环表示CP区域。"""

    def __init__(self):
        self.transformer = Transformer.from_crs(
            "EPSG:4978", "EPSG:4326", always_xy=True
        )
        self.trajectories = []

    def update(self, predictions: list[dict]) -> None:
        self.trajectories = [
            {
                "points": [
                    self.transformer.transform(*position)[:2]
                    for position in prediction["predicted_position_ecf"]
                ],
                "radii": prediction["radius_m"],
            }
            for prediction in predictions
        ]

    def draw(self, screen: pygame.Surface, renderer) -> None:
        for trajectory in self.trajectories:
            points = [
                renderer._geo_to_screen_projection(lon, lat)
                if renderer.use_projection
                else renderer._geo_to_screen_linear(lon, lat)
                for lon, lat in trajectory["points"]
            ]
            pygame.draw.lines(screen, (255, 235, 80), False, points, 2)
            for index in sorted(
                {0, len(points) // 4, len(points) // 2, 3 * len(points) // 4, len(points) - 1}
            ):
                radius = round(
                    trajectory["radii"][index]
                    * renderer.zoom
                    / (111320.0 * renderer.lat_per_pixel)
                )
                pygame.draw.circle(screen, (255, 165, 40), points[index], radius, 1)
                pygame.draw.circle(screen, (255, 235, 80), points[index], 3)


def _active_blue(simulator: InterceptorSimulator) -> bool:
    entity = simulator.entity_ext.entity
    return (
        simulator.launched == 1
        and entity.isVisible
        and entity.survivePoints > 0
        and simulator.ret < 0
        and not (entity.posEcf.x == 0 and entity.posEcf.y == 0 and entity.posEcf.z == 0)
    )


def _active_red(simulator) -> bool:
    entity = simulator.entity_ext.entity
    return (
        simulator.launch == 1
        and entity.isVisible
        and entity.survivePoints > 0
        and simulator.ret < 0
        and not (entity.posEcf.x == 0 and entity.posEcf.y == 0 and entity.posEcf.z == 0)
    )
