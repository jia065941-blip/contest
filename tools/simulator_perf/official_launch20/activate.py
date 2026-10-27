"""Enable launch20 acceleration before importing/constructing evaluation actors.

The SDK overlay must already be installed. Astra report ownership is separate
and opt-in; it is valid only for the frozen read-only member report pipeline.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
_enabled = False


def install(library, *, astra_reports=False):
    global _enabled
    if _enabled:
        raise RuntimeError('launch20 acceleration is already active')
    import envengine
    from envengine import fastcopy
    sdk = Path(envengine.__file__).resolve().parent.parent
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, hashes in manifest['files'].items():
        if hashlib.sha256((sdk / relative).read_bytes()).hexdigest() != hashes['after']:
            raise RuntimeError('Install the matching SDK overlay first: ' + relative)
    for relative, wanted in manifest['runtime_files'].items():
        if hashlib.sha256((HERE / relative).read_bytes()).hexdigest() != wanted:
            raise RuntimeError('Runtime source changed: ' + relative)
    spec = importlib.util.spec_from_file_location('astra_copy_core', library)
    compiled = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiled)
    old = fastcopy.deepcopy
    compiled.configure(old)
    for name, module in list(sys.modules.items()):
        if module is None or not (name == 'envengine' or name.startswith('envengine.')):
            continue
        for key, value in list(vars(module).items()):
            if value is old:
                setattr(module, key, compiled.deepcopy)
    from communication_copy import install as install_communications
    install_communications()
    if astra_reports:
        from ownership import install as install_reports
        install_reports()
    _enabled = True
    return {'sdk': str(sdk), 'library': str(library), 'astra_reports': astra_reports}
