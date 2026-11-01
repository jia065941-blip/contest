from pathlib import Path
import hashlib,json,os
from functools import lru_cache

HERE=Path(__file__).resolve().parent
HARNESS=HERE/'deps'
SDK_ROOT=HERE
ENV=SDK_ROOT/'.runtime/environment'
OLD=HERE/'deps/engine'
MODEL=HERE/'r3/model.pt'

def read(p):return json.loads(Path(p).read_text())

def read_hashed(p):
    data=Path(p).read_bytes()
    return json.loads(data),hashlib.sha256(data).hexdigest()

@lru_cache(maxsize=1)
def episode_inputs():
    """Read the inherited RAM file once, before any actor consumes its inputs.

    The disk fallback is only for inspecting/replaying the older file protocol.
    Neither path reads hidden simulator state or evaluation results.
    """
    fd=os.environ.get('MIGRATED_INPUT_FD')
    if fd is not None:
        with os.fdopen(int(fd),'r',encoding='utf-8') as stream:
            return json.load(stream)
    folder=Path(os.environ['MIGRATED_EPISODE'])
    return dict(compiled=read(folder/'plan.json'),opening=read(folder/'opening.json'))

def input_memfd(payload):
    """Anonymous Linux RAM storage, never a file on the cloud disk."""
    fd=os.memfd_create('astra-episode-input',os.MFD_CLOEXEC)
    try:
        data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        with os.fdopen(os.dup(fd),'wb') as stream:stream.write(data)
        os.lseek(fd,0,os.SEEK_SET)
        return fd
    except BaseException:
        os.close(fd);raise
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def verify_environment():
    info=read(SDK_ROOT/'.runtime/upstream.json')
    patches=read(SDK_ROOT/'optimization_manifest.json')['modified_sdk_files']
    for name,wanted in {**info['files'],**patches}.items():
        if sha(ENV/name)!=wanted:raise ValueError('New official environment changed: '+name)
    return dict(**{k:info[k] for k in ('repository','branch','commit','commit_time')}, optimization=read(SDK_ROOT/'optimization_manifest.json')['version'], patch_manifest_sha256=sha(SDK_ROOT/'optimization_manifest.json'))
