"""组织方场景预览：检查并初始化真实环境，输出二维图片和物理初态。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from envengine.environment.training_env import TrainingEnv
from envengine.render import Renderer
from scenarios import load_scenario


class Previewer:
    """组织方场景初态预览。"""

    def __init__(self, scenario: str | Path):
        self.scenario = Path(scenario).expanduser().resolve()

    def preview(self) -> tuple[bytes, dict]:
        """返回独立的 PNG 字节和物理状态，返回前关闭环境。"""
        scenario = load_scenario(self.scenario, include_rules=False)
        env = TrainingEnv(scenario, record_logs=False)
        try:
            state = env.reset()
            return Renderer.render(state), state
        finally:
            env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查场景并导出组织方二维预览及物理初态")
    parser.add_argument("scenario", type=Path, help="场景 JSON 文件路径")
    parser.add_argument("--output", type=Path, default=Path("preview"), help="输出根目录，按场景文件名建立子目录")
    args = parser.parse_args(argv)
    try:
        png, state = Previewer(args.scenario).preview()
        output = (args.output.expanduser() / args.scenario.stem).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "preview.png").write_bytes(png)
        (output / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False,
                       default=lambda value: value.to_dict()), encoding="utf-8",
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"预览失败：{error}\n")
    print(f"已预览 {len(state['entities'])} 个实体：{output / 'preview.png'}")
    print(f"物理初态：{output / 'state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
