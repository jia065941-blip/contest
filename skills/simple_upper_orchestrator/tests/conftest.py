from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT.parent
for path in (
    ROOT / "src",
    SKILLS / "staggered_deployment/src",
    SKILLS / "l_satellite_strike_packages/src",
    SKILLS / "hm_public_strike_packages/src",
    SKILLS / "local_avoidance/src",
    Path("/home/amax/ry/competition/competition-platform-env-release20261021"),
):
    sys.path.insert(0, str(path))
