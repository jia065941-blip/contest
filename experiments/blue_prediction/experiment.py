# -*-coding:utf-8 -*-
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

OBS_LEN = 10
PRED_LEN = 20
SPLIT_SEED = 42
TRAIN_RATIO = 0.6
CALIB_RATIO = 0.2
LSTM_HIDDEN = 32
LSTM_LR = 1e-3


def run_experiment(
    output_dir: str,
    scenario: str,
    rounds: int,
    max_steps: int,
    seed: int,
    epochs: int,
    batch_size: int,
    delta: float,
    side: str = "blue",
    pred_len: int = PRED_LEN,
) -> Path:
    from .collection import collect_trajectories

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    raw_dir = collect_trajectories(
        output / "raw",
        scenario,
        rounds,
        max_steps,
        seed,
        side,
    )
    dataset_dir = build_dataset([raw_dir], output / "dataset", side, pred_len)
    model, losses = LSTMPredictor.train(
        dataset_dir / "train.npz",
        output / "model",
        epochs,
        batch_size,
        seed,
    )
    _evaluate(
        output,
        dataset_dir,
        model,
        losses,
        {
            "scenario": scenario,
            "rounds": rounds,
            "max_steps": max_steps,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "delta": delta,
            "side": side,
            "obs_len": OBS_LEN,
            "pred_len": pred_len,
            "split_seed": SPLIT_SEED,
        },
    )
    return output


def reproduce_from_dataset(
    output_dir: str,
    dataset_dir: str,
    seed: int,
    epochs: int,
    batch_size: int,
    delta: float,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset = Path(dataset_dir)
    pred_len = int(np.load(dataset / "train.npz")["y"].shape[1])
    model, losses = LSTMPredictor.train(
        dataset / "train.npz",
        output / "model",
        epochs,
        batch_size,
        seed,
    )
    _evaluate(
        output,
        dataset,
        model,
        losses,
        {
            "source_dataset": str(dataset),
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "delta": delta,
            "obs_len": OBS_LEN,
            "pred_len": pred_len,
            "split_seed": SPLIT_SEED,
        },
    )
    return output


def build_dataset(
    run_dirs: list[Path],
    output_dir: Path,
    side: str,
    pred_len: int,
) -> Path:
    tracks, step_seconds = _load_tracks(run_dirs, side)
    split_keys = _split_tracks(tracks)
    output_dir.mkdir(parents=True, exist_ok=False)

    summary = {}
    for name, keys in split_keys.items():
        arrays = _build_windows(tracks, keys, step_seconds, side, pred_len)
        np.savez_compressed(output_dir / f"{name}.npz", **arrays)
        summary[name] = {
            "rounds": len({(key[0], key[1]) for key in keys}),
            "tracks": len(set(arrays["track_id"].tolist())),
            "windows": int(arrays["x"].shape[0]),
        }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source_run_dirs": [str(path) for path in run_dirs],
                "obs_len": OBS_LEN,
                "pred_len": pred_len,
                "side": side,
                "feature": f"{side}_entity_relative_ecef_position",
                "step_seconds": step_seconds,
                "split": {"train": 0.6, "calibration": 0.2, "test": 0.2},
                "split_unit": "complete_simulation_round",
                "split_seed": SPLIT_SEED,
                "summary": summary,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return output_dir


class LSTMPredictor:
    def __init__(self, model, x_mean, x_std, y_mean, y_std):
        self.model = model
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    @classmethod
    def train(
        cls,
        train_path: Path,
        model_dir: Path,
        epochs: int,
        batch_size: int,
        seed: int,
    ):
        model_dir.mkdir(parents=True, exist_ok=False)
        train = np.load(train_path)
        x, y = train["x"], train["y"]

        tf = _tensorflow()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()

        x_mean = x.mean(axis=(0, 1), keepdims=True)
        x_std = x.std(axis=(0, 1), keepdims=True)
        y_mean = y.mean(axis=(0, 1), keepdims=True)
        y_std = y.std(axis=(0, 1), keepdims=True)
        if np.any(x_std == 0) or np.any(y_std == 0):
            raise ValueError("训练数据存在零方差坐标，无法标准化")

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=x.shape[1:]),
                tf.keras.layers.LSTM(LSTM_HIDDEN),
                tf.keras.layers.Dense(y.shape[1] * 3),
                tf.keras.layers.Reshape((y.shape[1], 3)),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LR),
            loss="mean_squared_error",
        )
        history = model.fit(
            (x - x_mean) / x_std,
            (y - y_mean) / y_std,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=2,
        )
        model.save(model_dir / "model.keras")
        np.savez_compressed(
            model_dir / "normalization.npz",
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        )
        return cls(model, x_mean, x_std, y_mean, y_std), history.history["loss"]

    @classmethod
    def load(cls, model_dir: str | Path):
        model_dir = Path(model_dir)
        normalization = np.load(model_dir / "normalization.npz")
        return cls(
            _tensorflow().keras.models.load_model(
                model_dir / "model.keras", compile=False
            ),
            normalization["x_mean"],
            normalization["x_std"],
            normalization["y_mean"],
            normalization["y_std"],
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(
            (x - self.x_mean) / self.x_std,
            batch_size=256,
            verbose=0,
        )
        return prediction * self.y_std + self.y_mean


class ConformalCalibrator:
    def __init__(self, sigmas: np.ndarray, quantile: float, delta: float):
        self.sigmas = np.asarray(sigmas, dtype=np.float64)
        self.quantile = float(quantile)
        self.delta = float(delta)

    @classmethod
    def fit(cls, train_y, train_prediction, calib_y, calib_prediction, delta):
        if not 0 < delta < 1:
            raise ValueError("delta必须位于(0, 1)")
        sigmas = _errors(train_y, train_prediction).max(axis=0)
        if np.any(sigmas <= 0) or not np.all(np.isfinite(sigmas)):
            raise ValueError("CP归一化系数必须为有限正数")
        scores = np.max(
            _errors(calib_y, calib_prediction) / sigmas[np.newaxis, :],
            axis=1,
        )
        rank = int(np.ceil((len(scores) + 1) * (1 - delta)))
        if rank > len(scores):
            raise ValueError("校准样本不足，无法得到有限CP分位数")
        return cls(sigmas, np.sort(scores)[rank - 1], delta)

    @property
    def radii(self) -> np.ndarray:
        return self.sigmas * self.quantile

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "delta": self.delta,
                    "quantile": self.quantile,
                    "sigmas": self.sigmas.tolist(),
                    "radii": self.radii.tolist(),
                },
                file,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)
        return cls(data["sigmas"], data["quantile"], data["delta"])


def _evaluate(
    output: Path,
    dataset_dir: Path,
    model: LSTMPredictor,
    losses: list[float],
    config: dict,
) -> None:
    train = np.load(dataset_dir / "train.npz")
    calib = np.load(dataset_dir / "calib.npz")
    test = np.load(dataset_dir / "test.npz")

    predictions = {}
    calibrators = {}
    for name in ("cv", "lstm"):
        predictions[name] = {
            "train": _predict(name, train, model),
            "calib": _predict(name, calib, model),
            "test": _predict(name, test, model),
        }
        calibrators[name] = ConformalCalibrator.fit(
            train["y"],
            predictions[name]["train"],
            calib["y"],
            predictions[name]["calib"],
            config["delta"],
        )

    calibrators["lstm"].save(output / "cp.json")
    reports = {
        name: _model_report(
            test["y"],
            predictions[name]["test"],
            calibrators[name],
        )
        for name in ("cv", "lstm")
    }
    with (dataset_dir / "metadata.json").open(encoding="utf-8") as file:
        dataset_metadata = json.load(file)

    report = {
        "experiment": config,
        "dataset": dataset_metadata["summary"],
        "training_loss": losses,
        "models": reports,
        "comparison": {
            "lstm_ade_reduction_percent": _reduction(
                reports["cv"]["prediction"]["ade_m"],
                reports["lstm"]["prediction"]["ade_m"],
            ),
            "lstm_fde_reduction_percent": _reduction(
                reports["cv"]["prediction"]["fde_m"],
                reports["lstm"]["prediction"]["fde_m"],
            ),
        },
    }
    with (output / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "ade_m",
                "fde_m",
                "joint_coverage",
                "nominal_coverage",
                "mean_radius_m",
            ],
        )
        writer.writeheader()
        for name, result in reports.items():
            writer.writerow(
                {
                    "model": name,
                    "ade_m": result["prediction"]["ade_m"],
                    "fde_m": result["prediction"]["fde_m"],
                    "joint_coverage": result["conformal"]["joint_coverage"],
                    "nominal_coverage": result["nominal_coverage"],
                    "mean_radius_m": result["conformal"]["mean_radius_m"],
                }
            )

    np.savez_compressed(
        output / "predictions.npz",
        ground_truth=test["y"],
        cv_prediction=predictions["cv"]["test"],
        lstm_prediction=predictions["lstm"]["test"],
        lstm_radii=calibrators["lstm"].radii,
        track_id=test["track_id"],
    )


def _load_tracks(run_dirs: list[Path], side: str) -> tuple[dict, float]:
    tracks = defaultdict(list)
    step_seconds = None
    entity_id_field = "interceptor_id" if side == "blue" else "missile_id"
    for run_dir in run_dirs:
        with (run_dir / "metadata.json").open(encoding="utf-8") as file:
            current_step = float(json.load(file)["engine_step_ms"]) / 1000.0
        if step_seconds is None:
            step_seconds = current_step
        elif step_seconds != current_step:
            raise ValueError("所有采集数据必须使用相同引擎步长")

        run_id = run_dir.resolve().as_posix()
        for path in sorted(run_dir.glob("round_*.jsonl")):
            with path.open(encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        record = json.loads(line)
                        key = (
                            run_id,
                            int(record["round"]),
                            int(record[entity_id_field]),
                        )
                        tracks[key].append(record)
    if not tracks:
        raise ValueError("没有找到轨迹记录")
    for records in tracks.values():
        records.sort(key=lambda item: item["step"])
    return dict(tracks), step_seconds


def _split_tracks(tracks: dict) -> dict[str, list[tuple]]:
    rounds = sorted({(key[0], key[1]) for key in tracks})
    if len(rounds) < 3:
        raise ValueError("至少需要3个独立仿真回合")
    random.Random(SPLIT_SEED).shuffle(rounds)
    train_count = max(1, int(len(rounds) * TRAIN_RATIO))
    calib_count = max(1, int(len(rounds) * CALIB_RATIO))
    if train_count + calib_count >= len(rounds):
        train_count, calib_count = len(rounds) - 2, 1
    selected = {
        "train": set(rounds[:train_count]),
        "calib": set(rounds[train_count:train_count + calib_count]),
        "test": set(rounds[train_count + calib_count:]),
    }
    return {
        name: [key for key in tracks if (key[0], key[1]) in groups]
        for name, groups in selected.items()
    }


def _build_windows(
    tracks,
    keys,
    step_seconds,
    side,
    pred_len,
) -> dict[str, np.ndarray]:
    x, y, velocities, track_ids = [], [], [], []
    for key in keys:
        for segment in _continuous_segments(tracks[key], side):
            for start in range(len(segment) - OBS_LEN - pred_len + 1):
                window = segment[start:start + OBS_LEN + pred_len]
                history, future = window[:OBS_LEN], window[OBS_LEN:]
                reference = _position(history[-1], side)
                x.append([_position(record, side) - reference for record in history])
                y.append([_position(record, side) - reference for record in future])
                velocities.append(_velocity(history[-1], side))
                track_ids.append(f"{key[0]}|{key[1]}|{key[2]}")
    if not x:
        raise ValueError(f"没有长度达到 {OBS_LEN + pred_len} 的连续轨迹")
    return {
        "x": np.asarray(x, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "last_velocity": np.asarray(velocities, dtype=np.float32),
        "track_id": np.asarray(track_ids),
        "step_seconds": np.asarray(step_seconds, dtype=np.float32),
    }


def _continuous_segments(records: list[dict], side: str) -> list[list[dict]]:
    segments, current = [], []
    previous_step = previous_target = None
    for record in records:
        step = int(record["step"])
        target = (
            int(record["target_id"])
            if side == "blue"
            else int(record["entity_type"])
        )
        if previous_step is not None and (
            step != previous_step + 1 or target != previous_target
        ):
            segments.append(current)
            current = []
        current.append(record)
        previous_step, previous_target = step, target
    if current:
        segments.append(current)
    return segments


def _position(record: dict, side: str) -> np.ndarray:
    state_field = "interceptor" if side == "blue" else "missile"
    value = record[state_field]["position_ecf"]
    return np.asarray([value["x"], value["y"], value["z"]], dtype=np.float64)


def _velocity(record: dict, side: str) -> np.ndarray:
    state_field = "interceptor" if side == "blue" else "missile"
    value = record[state_field]["velocity_ecf"]
    return np.asarray([value["x"], value["y"], value["z"]], dtype=np.float64)


def _predict(name: str, dataset, model: LSTMPredictor) -> np.ndarray:
    if name == "lstm":
        return model.predict(dataset["x"])
    horizons = np.arange(1, dataset["y"].shape[1] + 1, dtype=np.float32)
    return (
        dataset["last_velocity"][:, np.newaxis, :]
        * horizons[np.newaxis, :, np.newaxis]
        * float(dataset["step_seconds"])
    )


def _errors(ground_truth, prediction) -> np.ndarray:
    if ground_truth.shape != prediction.shape:
        raise ValueError("真实轨迹与预测轨迹形状不一致")
    return np.linalg.norm(ground_truth - prediction, axis=-1)


def _model_report(ground_truth, prediction, calibrator) -> dict:
    errors = _errors(ground_truth, prediction)
    covered = errors <= calibrator.radii[np.newaxis, :]
    return {
        "prediction": {
            "ade_m": float(errors.mean()),
            "fde_m": float(errors[:, -1].mean()),
            "rmse_by_horizon_m": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
        },
        "nominal_coverage": 1 - calibrator.delta,
        "conformal": {
            "joint_coverage": float(np.all(covered, axis=1).mean()),
            "marginal_coverage": covered.mean(axis=0).tolist(),
            "mean_radius_m": float(calibrator.radii.mean()),
            "max_radius_m": float(calibrator.radii.max()),
            "radius_by_horizon_m": calibrator.radii.tolist(),
        },
    }


def _reduction(baseline: float, value: float) -> float:
    return float((baseline - value) / baseline * 100)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("需要安装 tensorflow") from error
    return tf
