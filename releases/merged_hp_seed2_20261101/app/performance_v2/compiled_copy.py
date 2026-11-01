"""Opt-in process-local copier; never modifies the sealed SDK on disk."""
from pathlib import Path
import hashlib, importlib.util, sys, sysconfig

HERE = Path(__file__).resolve().parent


def library_path():
    digest = hashlib.sha256((HERE/'copy_core.c').read_bytes()).hexdigest()[:16]
    return Path('/dev/shm/astra_copy_build')/(digest+sysconfig.get_config_var('EXT_SUFFIX'))


def load():
    from envengine import fastcopy
    path = library_path()
    spec = importlib.util.spec_from_file_location('astra_copy_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(fastcopy.deepcopy)
    return module


def install():
    from envengine import fastcopy
    old = fastcopy.deepcopy
    if getattr(old, '__module__', '') == 'astra_copy_core':
        raise RuntimeError('Compiled copier already installed')
    module = load()
    # Preserve all existing function bytecode; replace imported copier bindings
    # in SDK modules and cooperative adapters, not stdlib copy globally.
    changed = []
    for name, item in list(sys.modules.items()):
        if item is None or not name.startswith(('envengine', 'base_agents', 'agents')):
            continue
        for key, value in list(vars(item).items()):
            if value is old:
                setattr(item, key, module.deepcopy)
                changed.append(name+'.'+key)
    return dict(library=str(library_path()), bindings=changed)
