"""仿真程序启动入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from simulator import Simulator


def load_config(path: str | Path) -> dict:
    """读取公开运行配置，校验输入文件并统一解析相对路径。"""
    # 读取配置：展开用户目录并定位 YAML 文件，使用 safe_load 解析配置数据。
    path = Path(path).expanduser().resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {error}") from error

    # 检查公开字段：根节点必须是映射；除可选 seed 外，其余字段必填，不接受额外内部参数。
    required = {"agent", "scenario", "blue_mode", "render_mode", "output_dir", "logging"}
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must be a mapping")
    missing, unknown = required - config.keys(), config.keys() - required - {"seed"}
    if missing or unknown:
        raise ValueError(f"Configuration fields: missing={sorted(missing)}, unknown={sorted(map(str, unknown))}")

    # 校验基本选项：名称和路径使用非空字符串，渲染模式必须在支持范围内。
    for name in required - {"logging"}:
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty string")
    if config["render_mode"] not in {"human", "none"}:
        raise ValueError("render_mode must be human or none")

    # 解析输入文件：场景使用 JSON，双方策略使用 Python，均以 YAML 目录为基准。
    # 此处只检查文件路径，不读取场景内容、不导入或执行策略代码。
    for name, suffix in (("scenario", ".json"), ("blue_mode", ".py"), ("agent", ".py")):
        source = Path(config[name]).expanduser()
        source = (path.parent / source).resolve()
        if source.suffix.lower() != suffix:
            raise ValueError(f"{name} must reference a {suffix} file")
        if not source.is_file():
            raise ValueError(f"{name} must reference an existing file: {source}")
        config[name] = source

    # 校验随机种子：省略时保留 None，首次运行时生成；显式提供时必须是范围内的整数。
    # 使用 type 判断，避免将 Python 中属于 int 子类的布尔值误当作种子。
    config.setdefault("seed", None)
    if config["seed"] is not None and (type(config["seed"]) is not int or not 0 <= config["seed"] <= 2 ** 32 - 1):
        raise ValueError("seed must be null or an integer from 0 through 4294967295")

    # 校验输出配置：日志开关和示例评估开关使用布尔值，态势记录间隔使用正整数。
    # 该间隔只控制文件记录频率，不改变仿真步长或智能体的观测、决策频率。
    log = config["logging"]
    required_log = {"enabled", "state_interval_steps", "events", "actions"}
    if (not isinstance(log, dict) or not required_log <= log.keys()
            or log.keys() - required_log - {"example_evaluation", "attribution_evaluation"}):
        raise ValueError(
            "logging requires enabled, state_interval_steps, events and actions; "
            "example_evaluation and attribution_evaluation are optional")
    log.setdefault("example_evaluation", False)
    log.setdefault("attribution_evaluation", False)
    if any(type(log[name]) is not bool for name in (
        "enabled", "events", "actions", "example_evaluation", "attribution_evaluation",
    )):
        raise ValueError(
            "logging.enabled, events, actions, example_evaluation and "
            "attribution_evaluation must be booleans")
    if type(log["state_interval_steps"]) is not int or log["state_interval_steps"] <= 0:
        raise ValueError("logging.state_interval_steps must be a positive integer")

    # 处理输出目录：相对路径以 YAML 所在目录为基准，不受启动程序时的工作目录影响。
    output = Path(config["output_dir"]).expanduser()
    config["output_dir"] = (path.parent / output).resolve()
    return config

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Competition simulation")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"), help="Run configuration YAML file")
    args = parser.parse_args(argv)
    try:

        config = load_config(args.config)

        simulation = Simulator(config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    simulation.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
