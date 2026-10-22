"""Launch the release R9 adapter under the private glibc runtime when needed."""

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
    # Batch workers must not each create a full BLAS/OpenMP thread pool.
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    libc_name, libc_version = platform.libc_ver()
    needs_private = libc_name == "glibc" and _version(libc_version) < (2, 38)
    if needs_private and os.getenv("LOCAL_AVOID_RELEASE_PRIVATE_GLIBC") != "1":
        glibc = Path(os.getenv("LOCAL_AVOID_V13_GLIBC_ROOT", "/home/amax/ry/competition/glibc-2.38"))
        loader = glibc / "sysroot/lib64/ld-linux-x86-64.so.2"
        if not loader.is_file():
            raise RuntimeError(f"Private glibc loader is unavailable: {loader}")
        values = sys.argv[1:]
        try:
            root = Path(values[values.index("--competition-root") + 1]).resolve()
        except (ValueError, IndexError):
            root = Path("/nonexistent-competition-root")
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
        environment["LOCAL_AVOID_RELEASE_PRIVATE_GLIBC"] = "1"
        os.execve(loader, [
            str(loader), "--argv0", sys.executable, "--library-path", library_path,
            sys.executable, "-m", "local_avoid_v13_skill.release_r9_entry", *values,
        ], environment)
    from .release_r9_entry import main as entry_main
    return entry_main()


if __name__ == "__main__":
    raise SystemExit(main())
