"""Launch Plan Anchor replay under the private glibc runtime."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import sys


def main() -> int:
    libc_name, libc_version = platform.libc_ver()
    version = tuple(int(part) for part in libc_version.split(".") if part.isdigit())
    if (libc_name == "glibc" and version < (2, 38)
            and os.getenv("LOCAL_AVOID_PLAN_PRIVATE_GLIBC") != "1"):
        values = sys.argv[1:]
        try:
            plan_root = Path(values[values.index("--plan-root") + 1]).resolve()
        except (ValueError, IndexError):
            plan_root = Path("/nonexistent-plan-root")
        glibc = Path(os.getenv("LOCAL_AVOID_V13_GLIBC_ROOT",
                              "/home/amax/ry/competition/glibc-2.38"))
        loader = glibc / "sysroot/lib64/ld-linux-x86-64.so.2"
        native = plan_root / "core/envengine/simulator/models/HXDMissileModel"
        library_path = ":".join((
            str(glibc / "sysroot/lib64"), str(glibc / "sysroot/usr/lib64"),
            str(Path(sys.base_prefix) / "lib"), str(native),
            "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
        ))
        environment = dict(os.environ)
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG",
                     "LD_DEBUG_OUTPUT", "GLIBC_TUNABLES"):
            environment.pop(name, None)
        environment["LOCAL_AVOID_PLAN_PRIVATE_GLIBC"] = "1"
        os.execve(loader, [str(loader), "--argv0", sys.executable,
                           "--library-path", library_path, sys.executable,
                           "-m", "local_avoid_v13_skill.plan_anchor_entry", *values],
                  environment)
    from .plan_anchor_entry import main as entry_main
    return entry_main()


if __name__ == "__main__":
    raise SystemExit(main())
