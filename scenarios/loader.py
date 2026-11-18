"""解析本地场景路径，组装物理初态、部署约束与比赛规则。"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from .cases.reward import _reward_policy_from_scene
from .config import DeploymentConstraints, ScenarioConfig
from .deployment import resolve_deployment_constraints
from .scenario_checker import check_scenario


SCENARIO_ROOT = Path(__file__).resolve().parent
CASES_ROOT = SCENARIO_ROOT / "cases"


def resolve_scenario_path(scenario: str | Path = "default") -> Path | str:
    """将地图编号、JSON 路径或旧调用方使用的远程地址解析为来源。"""
    value = str(scenario)
    if value.startswith(("http://", "https://")):
        return value
    if value == "default":
        return SCENARIO_ROOT / "E01.json"

    external = Path(value).expanduser()
    if external.suffix.lower() == ".json" and external.is_file():
        return external.resolve()
    # 项目内的完整地图路径不依赖调用方工作目录。
    normalized = value.replace("\\", "/").removeprefix("./")
    if normalized.startswith("scenarios/"):
        local = (SCENARIO_ROOT / normalized.removeprefix("scenarios/")).resolve()
        if local.is_relative_to(SCENARIO_ROOT) and local.suffix.lower() == ".json" and local.is_file():
            return local

    # 编号与文件名直接对应，例如 E01 和 E01.json 都定位到场景目录。
    if "/" not in normalized:
        filename = normalized if external.suffix.lower() == ".json" else normalized + ".json"
        bundled = SCENARIO_ROOT / filename
        if bundled.is_file():
            return bundled
    raise ValueError(
        f"Unknown scenario '{value}'. Use e.g. E01, scenarios/E01.json, "
        "or an existing JSON file path."
    )


def load_scenario(scenario: str | Path = "default", *, include_rules: bool = True) -> ScenarioConfig:
    """单次读取本地场景并组装独立配置，预览时可省略计分规则。"""
    path = resolve_scenario_path(scenario)
    if isinstance(path, str):
        raise ValueError("场景配置加载仅支持本地 JSON 文件")
    path = path.resolve()

    # 场景只读取一次，物理初态、地图说明和评分目标使用同一份数据。
    data = check_scenario(path)
    from envengine.environment.scene_adapter import profile_from_scene

    profile = profile_from_scene(data)
    coordinates, launch_altitude = resolve_deployment_constraints(profile)

    # 场景时长决定唯一时限，计分规则与蓝方价值仅在运行加载时构造。
    imagine = profile.imagineProfile
    max_steps = int(imagine.simEndLogicTime // imagine.simStep)
    reward_policy = _reward_policy_from_scene(path, data, max_steps) if include_rules else None
    return ScenarioConfig(
        path=path,
        profile=profile,
        deployment=DeploymentConstraints(coordinates, launch_altitude),
        reward_policy=reward_policy,
    )


def read_profile(source: str):
    """为直接调用方保留本地文件和 HTTP(S) 来源的 Profile 读取。"""
    from envengine.environment.scene_adapter import profile_from_scene

    if urlparse(source).scheme in {"http", "https"}:
        import requests

        response = requests.get(source, timeout=30)
        response.raise_for_status()
        data = response.json()
        return profile_from_scene(data.get("data", data))
    path = Path(source)
    if not path.is_file():
        raise ValueError(f"Scenario file does not exist: {source}")
    return profile_from_scene(json.loads(path.read_text(encoding="utf-8")))
