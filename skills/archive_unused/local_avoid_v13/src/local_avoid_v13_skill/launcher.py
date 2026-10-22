"""Re-execute the adapter under the private glibc loader when required."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import sys


def _version(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def main() -> int:
    libc_name, libc_version = platform.libc_ver()
    needs_private = libc_name == "glibc" and _version(libc_version) < (2, 38)
    if needs_private and os.getenv("LOCAL_AVOID_V13_PRIVATE_GLIBC") != "1":
        glibc = Path(os.getenv("LOCAL_AVOID_V13_GLIBC_ROOT",
                              "/home/amax/ry/competition/glibc-2.38"))
        loader = glibc / "sysroot/lib64/ld-linux-x86-64.so.2"
        if not loader.is_file():
            raise RuntimeError(f"Private glibc loader is unavailable: {loader}")
        values = sys.argv[1:]
        try:
            root_index = values.index("--competition-root") + 1
            competition_root = Path(values[root_index]).resolve()
        except (ValueError, IndexError):
            competition_root = Path("/nonexistent-competition-root")
        native = competition_root / "core/envengine/simulator/models/HXDMissileModel"
        conda_lib = Path(sys.base_prefix) / "lib"
        library_path = ":".join((
            str(glibc / "sysroot/lib64"), str(glibc / "sysroot/usr/lib64"),
            str(conda_lib), str(native), "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
        ))
        environment = dict(os.environ)
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG",
                     "LD_DEBUG_OUTPUT", "GLIBC_TUNABLES"):
            environment.pop(name, None)
        environment["LOCAL_AVOID_V13_PRIVATE_GLIBC"] = "1"
        os.execve(loader, [str(loader), "--argv0", sys.executable,
                           "--library-path", library_path, sys.executable,
                           "-m", "local_avoid_v13_skill.official_entry", *values], environment)
    from .official_entry import main as official_main
    return official_main()


if __name__ == "__main__":
    raise SystemExit(main())
