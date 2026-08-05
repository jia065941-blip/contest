# -*-coding:utf-8 -*-
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import main as platform_main
import numpy as np
import pygame
from pyproj import Transformer

from envengine import TrainingEnv
from envengine.simulator.simlulator_impl.InterceptorSimulator import (
    InterceptorSimulator,
)
from trajectory_prediction.collection import _register_agents
from main import read_profile

RED_ENTITY_TYPES = (21000, 21001, 21002)


@dataclass(frozen=True)
class IntentParameters:
    heading_weight: float
    closing_weight: float
    cpa_scale_m: float
    acceleration_scale_mps2: float
    likelihood_floor: float = 1e-4
    bayes_smoothing: float = 0.05
    navigation_constant: float = 3.0

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as file:
            return cls(**json.load(file)["parameters"])


class OnlineIntentPredictor:
    """根据相对运动关系递推估计每枚蓝方拦截弹的目标。"""

    def __init__(self, parameters_path):
        self.parameters = IntentParameters.load(parameters_path)
        self.posteriors = {}
        self.candidate_sets = {}
        self.previous_velocities = {}
        self.first_steps = {}
        self.round_id = None
        self.total = 0
        self.correct = 0
        self.top3_correct = 0
        self.mature_total = 0
        self.mature_correct = 0
        self.confident_total = 0
        self.confident_correct = 0
        self.random_accuracy_sum = 0.0
        self.latest = {}

    def update(self, env) -> list[dict]:
        if self.round_id != env.current_round:
            self.posteriors.clear()
            self.candidate_sets.clear()
            self.previous_velocities.clear()
            self.first_steps.clear()
            self.round_id = env.current_round

        red_entities = _detected_red_entities(env)
        if not red_entities:
            return []

        results, active_ids = [], set()
        factory = env.engine.simulator_factory
        for simulator in factory.get_simulators_by_type(24000):
            if not isinstance(simulator, InterceptorSimulator):
                raise TypeError(f"实体 {simulator.entity_ext.entity.id} 不是拦截弹仿真器")
            if not _active_interceptor(simulator):
                continue
            actual_target = factory.get_simulator_by_id(simulator.target_id)
            if actual_target is None or not _alive(actual_target.entity_ext.entity):
                continue

            entity = simulator.entity_ext.entity
            interceptor_id = int(entity.id)
            active_ids.add(interceptor_id)
            self.first_steps.setdefault(interceptor_id, int(env.current_step))
            self.candidate_sets.setdefault(interceptor_id, tuple(red_entities))
            candidate_ids = np.asarray(
                [
                    candidate_id
                    for candidate_id in self.candidate_sets[interceptor_id]
                    if candidate_id in red_entities
                ]
            )
            candidates = [red_entities[int(candidate_id)] for candidate_id in candidate_ids]
            candidate_positions = np.stack(
                [_vector(candidate.posEcf) for candidate in candidates]
            )
            candidate_velocities = np.stack(
                [_vector(candidate.velEcf) for candidate in candidates]
            )
            interceptor_velocity = _vector(entity.velEcf)
            previous_velocity = self.previous_velocities.get(interceptor_id)
            self.previous_velocities[interceptor_id] = interceptor_velocity
            if previous_velocity is None:
                continue
            acceleration = (
                interceptor_velocity - previous_velocity
            ) / (env.engine.sim_step / 1000.0)
            posterior = self._update_posterior(
                interceptor_id,
                candidate_ids,
                candidate_positions,
                candidate_velocities,
                _vector(entity.posEcf),
                interceptor_velocity,
                acceleration,
            )
            best_index = int(np.argmax(posterior))
            predicted_target_id = int(candidate_ids[best_index])
            actual_target_id = int(simulator.target_id)
            correct = predicted_target_id == actual_target_id
            top3_ids = candidate_ids[np.argsort(posterior)[-3:][::-1]]
            top3_correct = actual_target_id in top3_ids
            age_steps = int(env.current_step) - self.first_steps[interceptor_id]
            confidence = float(posterior[best_index])
            self.total += 1
            self.correct += int(correct)
            self.top3_correct += int(top3_correct)
            self.random_accuracy_sum += 1.0 / len(candidate_ids)
            if age_steps >= 10:
                self.mature_total += 1
                self.mature_correct += int(correct)
            if confidence >= 0.5:
                self.confident_total += 1
                self.confident_correct += int(correct)
            self.latest[(int(env.current_round), interceptor_id)] = correct
            results.append(
                {
                    "interceptor_id": interceptor_id,
                    "predicted_target_id": predicted_target_id,
                    "actual_target_id": actual_target_id,
                    "confidence": confidence,
                    "correct": correct,
                    "top3_target_ids": [int(candidate_id) for candidate_id in top3_ids],
                    "age_steps": age_steps,
                    "probabilities": {
                        str(int(candidate_id)): float(probability)
                        for candidate_id, probability in zip(candidate_ids, posterior)
                    },
                    "interceptor_position_ecf": _vector(entity.posEcf).tolist(),
                    "target_position_ecf": candidate_positions[best_index].tolist(),
                }
            )

        for interceptor_id in self.posteriors.keys() - active_ids:
            del self.posteriors[interceptor_id]
        return results

    def _update_posterior(
        self,
        interceptor_id,
        candidate_ids,
        candidate_positions,
        candidate_velocities,
        interceptor_position,
        interceptor_velocity,
        interceptor_acceleration,
    ):
        features = _intent_features(
            candidate_positions,
            candidate_velocities,
            interceptor_position,
            interceptor_velocity,
            interceptor_acceleration,
            self.parameters.navigation_constant,
        )
        heading_alignment, closing_score, cpa_distance, acceleration_error = features.T
        parameters = self.parameters
        log_score = (
            parameters.heading_weight * heading_alignment
            + parameters.closing_weight * closing_score
            - cpa_distance / parameters.cpa_scale_m
            - acceleration_error / parameters.acceleration_scale_mps2
        )
        likelihood = np.exp(log_score - np.max(log_score))
        likelihood = np.maximum(likelihood, parameters.likelihood_floor)

        previous = self.posteriors.get(interceptor_id, {})
        prior = np.asarray(
            [previous.get(int(candidate_id), 1.0) for candidate_id in candidate_ids],
            dtype=np.float64,
        )
        prior /= prior.sum()
        bayes = prior * likelihood
        bayes /= bayes.sum()
        posterior = (
            parameters.bayes_smoothing * prior
            + (1.0 - parameters.bayes_smoothing) * bayes
        )
        posterior /= posterior.sum()
        self.posteriors[interceptor_id] = {
            int(candidate_id): float(probability)
            for candidate_id, probability in zip(candidate_ids, posterior)
        }
        return posterior

    def summary(self) -> dict:
        return {
            "samples": self.total,
            "random_top1_accuracy": (
                self.random_accuracy_sum / self.total if self.total else 0.0
            ),
            "top1_accuracy": self.correct / self.total if self.total else 0.0,
            "top3_accuracy": self.top3_correct / self.total if self.total else 0.0,
            "top1_accuracy_after_10_steps": (
                self.mature_correct / self.mature_total if self.mature_total else 0.0
            ),
            "confidence_0_5_coverage": (
                self.confident_total / self.total if self.total else 0.0
            ),
            "confidence_0_5_accuracy": (
                self.confident_correct / self.confident_total
                if self.confident_total
                else 0.0
            ),
            "final_top1_accuracy": (
                sum(self.latest.values()) / len(self.latest) if self.latest else 0.0
            ),
            "parameters": asdict(self.parameters),
            "candidate_source": "defend_commander_detect_info_at_first_observation",
        }


class IntentCalibrationCollector:
    """采集候选目标特征，并用负对数似然自动标定参数。"""

    def __init__(self):
        self.candidate_sets = {}
        self.previous_velocities = {}
        self.samples = []
        self.round_id = None

    def update(self, env):
        if self.round_id != env.current_round:
            self.candidate_sets.clear()
            self.previous_velocities.clear()
            self.round_id = env.current_round

        red_entities = _detected_red_entities(env)
        factory = env.engine.simulator_factory
        for simulator in factory.get_simulators_by_type(24000):
            if not _active_interceptor(simulator):
                continue
            actual_target = factory.get_simulator_by_id(simulator.target_id)
            if actual_target is None or not _alive(actual_target.entity_ext.entity):
                continue

            entity = simulator.entity_ext.entity
            interceptor_id = int(entity.id)
            self.candidate_sets.setdefault(interceptor_id, tuple(red_entities))
            candidate_ids = np.asarray(
                [
                    candidate_id
                    for candidate_id in self.candidate_sets[interceptor_id]
                    if candidate_id in red_entities
                ]
            )
            actual_indices = np.flatnonzero(candidate_ids == int(simulator.target_id))
            if len(actual_indices) != 1:
                raise ValueError(f"拦截弹 {interceptor_id} 的真实目标不在固定候选集中")

            interceptor_velocity = _vector(entity.velEcf)
            previous_velocity = self.previous_velocities.get(interceptor_id)
            self.previous_velocities[interceptor_id] = interceptor_velocity
            if previous_velocity is None:
                continue
            candidates = [red_entities[int(candidate_id)] for candidate_id in candidate_ids]
            features = _intent_features(
                np.stack([_vector(candidate.posEcf) for candidate in candidates]),
                np.stack([_vector(candidate.velEcf) for candidate in candidates]),
                _vector(entity.posEcf),
                interceptor_velocity,
                (interceptor_velocity - previous_velocity)
                / (env.engine.sim_step / 1000.0),
                3.0,
            )
            self.samples.append((features, int(actual_indices[0])))

    def fit(self):
        if not self.samples:
            raise ValueError("没有采集到可用于意图参数标定的样本")
        best_loss, best_parameters = float("inf"), None
        for heading_weight in (2.0, 4.0, 6.0):
            for closing_weight in (0.5, 1.0, 2.0):
                for cpa_scale_m in (10000.0, 20000.0, 40000.0):
                    for acceleration_scale_mps2 in (20.0, 50.0, 100.0, 200.0):
                        parameters = IntentParameters(
                            heading_weight,
                            closing_weight,
                            cpa_scale_m,
                            acceleration_scale_mps2,
                        )
                        loss = _mean_negative_log_likelihood(
                            self.samples, parameters
                        )
                        if loss < best_loss:
                            best_loss = loss
                            best_parameters = parameters
        return best_parameters, best_loss


def calibrate_intent_parameters(
    output_dir,
    scenario="./scenarios/platform.json",
    rounds=3,
    max_steps=120,
    seed=1,
):
    random.seed(seed)
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    profile = read_profile(scenario)
    env = TrainingEnv(profile, render_mode="none")
    _register_agents(env, profile)
    env.reset()
    collector = IntentCalibrationCollector()
    try:
        for round_index in range(rounds):
            env.red_model_deploy()
            for _ in range(max_steps):
                _, _, done, _ = env.step()
                collector.update(env)
                if done:
                    break
            if round_index + 1 < rounds:
                env.reset()
    finally:
        env.close()

    parameters, calibration_nll = collector.fit()
    result = {
        "scenario": scenario,
        "rounds": rounds,
        "max_steps": max_steps,
        "seed": seed,
        "samples": len(collector.samples),
        "calibration_nll": calibration_nll,
        "candidate_source": "defend_commander_detect_info_at_first_observation",
        "parameters": asdict(parameters),
    }
    path = output_path / "calibration.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return path


class IntentOverlay:
    def __init__(self):
        self.transformer = Transformer.from_crs(
            "EPSG:4978", "EPSG:4326", always_xy=True
        )
        self.items = []
        self.summary = {}
        self.font = None

    def update(self, predictions, summary):
        ranked = sorted(
            predictions,
            key=lambda item: item["confidence"],
            reverse=True,
        )[:30]
        self.items = [
            {
                "interceptor": self.transformer.transform(
                    *item["interceptor_position_ecf"]
                )[:2],
                "target": self.transformer.transform(*item["target_position_ecf"])[:2],
                "target_id": item["predicted_target_id"],
                "confidence": item["confidence"],
                "correct": item["correct"],
            }
            for item in ranked
        ]
        self.summary = summary

    def draw(self, screen, renderer):
        if self.font is None:
            self.font = pygame.font.Font(None, 18)
        for item in self.items:
            start = _screen_point(renderer, item["interceptor"])
            end = _screen_point(renderer, item["target"])
            color = (80, 220, 120) if item["correct"] else (240, 90, 90)
            _draw_dashed_line(screen, color, start, end)
            pygame.draw.circle(screen, color, end, 6, 2)
            label = self.font.render(
                f"{item['target_id']} {item['confidence']:.2f}",
                True,
                color,
            )
            screen.blit(label, ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2))

        text = self.font.render(
            f"Intent Top-1: {self.summary.get('top1_accuracy', 0.0):.1%}  "
            f"Samples: {self.summary.get('samples', 0)}",
            True,
            (255, 255, 255),
        )
        screen.blit(text, (12, 12))


def run_main_with_intent_prediction(
    parameters_path="results/interceptor_intent/calibration.json",
) -> None:
    output_dir = Path("results/interceptor_intent_online")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{datetime.now():%Y%m%d%H%M%S}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")

    predictor = OnlineIntentPredictor(parameters_path)
    overlay = IntentOverlay()
    active_renderer = {"value": None}
    original_step = TrainingEnv.step
    original_flip = pygame.display.flip

    with output_path.open("x", encoding="utf-8") as output_file:
        def flip_with_intent():
            renderer = active_renderer["value"]
            screen = pygame.display.get_surface()
            if renderer is not None and screen is not None:
                overlay.draw(screen, renderer)
            original_flip()

        def step_with_intent(env, *args, **kwargs):
            result = original_step(env, *args, **kwargs)
            predictions = predictor.update(env)
            env.interceptor_intent_predictions = predictions
            if env.renderer is not None:
                active_renderer["value"] = env.renderer
                overlay.update(predictions, predictor.summary())
            output_file.write(
                json.dumps(
                    {
                        "round": env.current_round,
                        "step": env.current_step,
                        "predictions": predictions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            return result

        TrainingEnv.step = step_with_intent
        pygame.display.flip = flip_with_intent
        try:
            platform_main.main()
        finally:
            TrainingEnv.step = original_step
            pygame.display.flip = original_flip

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(predictor.summary(), file, ensure_ascii=False, indent=2)


def _intent_features(
    candidate_positions,
    candidate_velocities,
    interceptor_position,
    interceptor_velocity,
    interceptor_acceleration,
    navigation_constant,
):
    relative_position = candidate_positions - interceptor_position
    ranges = np.linalg.norm(relative_position, axis=1)
    line_of_sight = relative_position / ranges[:, np.newaxis]
    speed = np.linalg.norm(interceptor_velocity)
    velocity_direction = interceptor_velocity / speed
    heading_alignment = line_of_sight @ velocity_direction

    relative_velocity = candidate_velocities - interceptor_velocity
    closing_speed = -np.sum(line_of_sight * relative_velocity, axis=1)
    velocity_square = np.sum(relative_velocity**2, axis=1)
    time_to_cpa = np.maximum(
        0.0,
        -np.sum(relative_position * relative_velocity, axis=1) / velocity_square,
    )
    cpa_position = relative_position + time_to_cpa[:, np.newaxis] * relative_velocity
    cpa_distance = np.linalg.norm(cpa_position, axis=1)

    los_rate = np.cross(relative_position, relative_velocity) / (
        ranges[:, np.newaxis] ** 2
    )
    pn_acceleration = (
        navigation_constant
        * closing_speed[:, np.newaxis]
        * np.cross(los_rate, velocity_direction)
    )
    normal_acceleration = interceptor_acceleration - (
        np.dot(interceptor_acceleration, velocity_direction) * velocity_direction
    )
    acceleration_error = np.linalg.norm(
        pn_acceleration - normal_acceleration,
        axis=1,
    )
    return np.column_stack(
        (
            heading_alignment,
            np.tanh(closing_speed / 500.0),
            cpa_distance,
            acceleration_error,
        )
    )


def _mean_negative_log_likelihood(samples, parameters):
    losses = []
    for features, actual_index in samples:
        heading_alignment, closing_score, cpa_distance, acceleration_error = features.T
        scores = (
            parameters.heading_weight * heading_alignment
            + parameters.closing_weight * closing_score
            - cpa_distance / parameters.cpa_scale_m
            - acceleration_error / parameters.acceleration_scale_mps2
        )
        maximum = np.max(scores)
        log_normalizer = maximum + np.log(np.exp(scores - maximum).sum())
        losses.append(log_normalizer - scores[actual_index])
    return float(np.mean(losses))


def _active_red_entities(env):
    factory = env.engine.simulator_factory
    return [
        simulator.entity_ext.entity
        for entity_type in RED_ENTITY_TYPES
        for simulator in factory.get_simulators_by_type(entity_type)
        if getattr(simulator, "launch", 0) == 1
        and getattr(simulator, "ret", 0) < 0
        and _alive(simulator.entity_ext.entity)
        and np.linalg.norm(_vector(simulator.entity_ext.entity.posEcf)) > 0
    ]


def _detected_red_entities(env):
    active_entities = {
        int(entity.id): entity for entity in _active_red_entities(env)
    }
    detected_ids = {
        int(entity_id)
        for commander in env.engine.simulator_factory.get_simulators_by_type(35000)
        for entity_id in commander.entity_ext.entity.detectInfo
    }
    return {
        entity_id: entity
        for entity_id, entity in active_entities.items()
        if entity_id in detected_ids
    }


def _active_interceptor(simulator):
    entity = simulator.entity_ext.entity
    return (
        simulator.launched == 1
        and simulator.target_id is not None
        and simulator.ret < 0
        and _alive(entity)
        and np.linalg.norm(_vector(entity.velEcf)) > 0
    )


def _screen_point(renderer, lon_lat):
    if renderer.use_projection:
        return renderer._geo_to_screen_projection(*lon_lat)
    return renderer._geo_to_screen_linear(*lon_lat)


def _draw_dashed_line(screen, color, start, end):
    vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length == 0:
        return
    direction = vector / length
    for offset in np.arange(0, length, 12):
        segment_start = np.asarray(start) + direction * offset
        segment_end = np.asarray(start) + direction * min(offset + 7, length)
        pygame.draw.line(screen, color, segment_start, segment_end, 1)


def _alive(entity):
    return bool(entity.isVisible and entity.survivePoints > 0)


def _vector(value):
    return np.asarray([value.x, value.y, value.z], dtype=np.float64)
