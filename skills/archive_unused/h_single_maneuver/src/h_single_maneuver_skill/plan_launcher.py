"""Launch PlanSpec evaluation under the private glibc runtime when needed."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import sys


def main() -> int:
    libc_name, libc_version = platform.libc_ver()
    version = tuple(int(part) for part in libc_version.split(".") if part.isdigit())
    if (
        (libc_name == "glibc" and version < (2, 38))
        or sys.version_info[:2] != (3, 11)
    ) and os.getenv("H_SINGLE_MANEUVER_PRIVATE_GLIBC") != "1":
        values = sys.argv[1:]
        try:
            environment_root = Path(
                values[values.index("--environment-root") + 1]
            ).resolve()
        except (ValueError, IndexError):
            environment_root = Path("/nonexistent-environment-root")
        glibc = Path(
            os.getenv(
                "H_SINGLE_MANEUVER_GLIBC_ROOT",
                "/home/amax/ry/competition/glibc-2.38",
            )
        )
        loader = glibc / "sysroot/lib64/ld-linux-x86-64.so.2"
        environment_core = (
            environment_root / "core"
            if (environment_root / "core/envengine").is_dir()
            else environment_root
        )
        native = environment_core / "envengine/simulator/models/HXDMissileModel"
        python = Path(
            os.getenv(
                "H_SINGLE_MANEUVER_PYTHON",
                "/home/amax/ry/competition/personal_train/.conda/competition-rl/bin/python",
            )
        ).resolve()
        source = Path(__file__).resolve().parents[1]
        skill_root = Path(__file__).resolve().parents[4]
        library_path = ":".join(
            (
                str(glibc / "sysroot/lib64"),
                str(glibc / "sysroot/usr/lib64"),
                str(python.parent.parent / "lib"),
                str(native),
                "/lib/x86_64-linux-gnu",
                "/usr/lib/x86_64-linux-gnu",
            )
        )
        environment = dict(os.environ)
        for name in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "LD_AUDIT",
            "LD_DEBUG",
            "LD_DEBUG_OUTPUT",
            "GLIBC_TUNABLES",
        ):
            environment.pop(name, None)
        environment["H_SINGLE_MANEUVER_PRIVATE_GLIBC"] = "1"
        source_paths = (
            glibc / "runtime/python-packages",
            source,
            skill_root / "src",
            skill_root / "skills/turn_left/src",
            skill_root / "skills/turn_right/src",
            skill_root / "skills/stop_turning/src",
            skill_root / "skills/three_missile_joint/src",
            skill_root / "skills/local_avoid_v13/src",
            skill_root / "skills/l_search_coordination/src",
            skill_root / "skills/l_response_package/src",
            skill_root / "skills/target_damage_belief/src",
            skill_root / "skills/mission_replan/src",
            skill_root / "skills/m_strike_package/src",
        )
        environment["PYTHONPATH"] = ":".join(
            [str(path) for path in source_paths]
            + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        os.execve(
            loader,
            [
                str(loader),
                "--argv0",
                str(python),
                "--library-path",
                library_path,
                str(python),
                "-m",
                "h_single_maneuver_skill.plan_entry",
                *values,
            ],
            environment,
        )
    from .plan_entry import main as entry_main

    return entry_main()


if __name__ == "__main__":
    raise SystemExit(main())
