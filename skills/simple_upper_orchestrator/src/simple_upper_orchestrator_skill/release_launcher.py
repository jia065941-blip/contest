"""Re-exec the adapter under a caller-selectable private glibc runtime."""

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


def _argument(name: str, default: str) -> Path:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1]).expanduser().resolve()
    except (ValueError, IndexError):
        return Path(default).expanduser().resolve()


def main() -> int:
    threads = os.getenv("SIMPLE_UPPER_THREADS", "1")
    try:
        if int(threads) <= 0:
            raise ValueError
    except ValueError as error:
        raise RuntimeError("SIMPLE_UPPER_THREADS must be a positive integer") from error
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS", "TORCH_NUM_THREADS",
    ):
        os.environ[name] = threads
    libc_name, libc_version = platform.libc_ver()
    needs_private = libc_name == "glibc" and _version(libc_version) < (2, 38)
    marker = "SIMPLE_UPPER_PRIVATE_GLIBC"
    if needs_private and os.getenv(marker) != "1":
        glibc = Path(os.getenv("SIMPLE_UPPER_GLIBC_ROOT", "/home/amax/ry/competition/glibc-2.38"))
        loader = glibc / "sysroot/lib64/ld-linux-x86-64.so.2"
        if not loader.is_file():
            raise RuntimeError(f"private glibc loader is unavailable: {loader}")
        root = _argument("--competition-root", "/home/amax/ry/competition/competition-platform-env-release20261021")
        conda_lib = Path(sys.base_prefix) / "lib"
        native = root / "envengine/simulator/models/HXDMissileModel"
        library_path = ":".join((
            str(glibc / "sysroot/lib64"), str(glibc / "sysroot/usr/lib64"),
            str(conda_lib), str(native), str(root),
            "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
        ))
        environment = dict(os.environ)
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "LD_DEBUG_OUTPUT", "GLIBC_TUNABLES"):
            environment.pop(name, None)
        environment[marker] = "1"
        os.execve(loader, [
            str(loader), "--argv0", sys.executable, "--library-path", library_path,
            sys.executable, "-m", "simple_upper_orchestrator_skill.release_entry", *sys.argv[1:],
        ], environment)
    from .release_entry import main as entry_main
    return entry_main()


if __name__ == "__main__":
    raise SystemExit(main())
