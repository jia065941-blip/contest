"""Keep the release's rebuilt copier out of historical experiment caches."""
from pathlib import Path
import hashlib
import sysconfig

def configure():
    import compiled_copy
    source=Path(__file__).resolve().parent/'performance_v2/copy_core.c'
    name=hashlib.sha256(source.read_bytes()).hexdigest()[:16]+sysconfig.get_config_var('EXT_SUFFIX')
    path=Path('/dev/shm/merged_hp_release_copy')/name
    compiled_copy.library_path=lambda:path
    return path
