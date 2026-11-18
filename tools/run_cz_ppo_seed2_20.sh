#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
RELEASE=${CZ_RELEASE:-$ROOT/vendor/cz_merged_hp_seed2_20261101}
PYTHON=${PYTHON:-python3}

export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CZ_RELEASE=$RELEASE

exec "$PYTHON" -B "$ROOT/tools/run_cz_ppo_seed2_20.py" "$@"
