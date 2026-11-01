"""Install the bundled reference Python and locked dependencies; no simulations."""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
if sys.flags.optimize:
    raise RuntimeError('Optimized Python is not permitted for installation validation')

class PartReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths=iter(paths);self.stream=None
    def readable(self):return True
    def readinto(self, buffer):
        while True:
            if self.stream is None:
                path=next(self.paths,None)
                if path is None:return 0
                self.stream=path.open('rb')
            size=self.stream.readinto(buffer)
            if size:return size
            self.stream.close();self.stream=None
    def close(self):
        if self.stream is not None:self.stream.close()
        super().close()

def install_packages(venv):
    manifest=json.loads((ROOT/'dependencies_manifest.json').read_text())
    paths=[]
    for part in manifest['parts']:
        path=ROOT/part['path']
        if path.stat().st_size!=part['bytes'] or sha(path)!=part['sha256']:
            raise RuntimeError('Dependency archive part changed: '+part['path'])
        paths.append(path)
    target=venv/'lib/python3.11/site-packages';seen=set()
    with PartReader(paths) as raw, tarfile.open(fileobj=raw,mode='r|gz') as tar:
        for member in tar:
            name=PurePosixPath(member.name);p=target/member.name
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.name in seen:
                raise RuntimeError('Unsafe or duplicate dependency member')
            if member.name not in manifest['files']:raise RuntimeError('Unexpected dependency file')
            seen.add(member.name);expected=manifest['files'][member.name]
            if any(parent.is_symlink() for parent in [p,*p.parents]):raise RuntimeError('Symlink in dependency destination')
            if not p.exists():
                p.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,p.open('xb') as output:
                    while chunk:=source.read(1024*1024):output.write(chunk)
            if sha(p)!=expected['sha256']:raise RuntimeError('Dependency file differs: '+member.name)
            p.chmod(expected['mode'])
    if seen!=set(manifest['files']):raise RuntimeError('Incomplete dependency archive')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Linux x86_64 is required')
    reference=json.loads((ROOT/'python_manifest.json').read_text())
    archive=ROOT/'python-runtime.tar.gz';target=ROOT/'.python'
    if sha(archive)!=reference['archive_sha256']:raise RuntimeError('Python archive changed')
    if target.is_symlink():raise RuntimeError('Symlink at Python destination')
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        expected=set(reference['files'])|set(reference['symlinks'])
        if len(members)!=len(expected) or {m.name for m in members}!=expected:
            raise RuntimeError('Python archive inventory mismatch')
        for member in members:
            name=PurePosixPath(member.name);p=target/member.name
            if name.is_absolute() or '..' in name.parts:raise RuntimeError('Unsafe Python member')
            if any(parent.is_symlink() for parent in p.parents):raise RuntimeError('Symlink parent')
            p.parent.mkdir(parents=True,exist_ok=True)
            if member.isfile():
                if p.is_symlink():raise RuntimeError('Unexpected file symlink')
                if not p.exists():p.write_bytes(tar.extractfile(member).read())
                if sha(p)!=reference['files'][member.name]['sha256']:raise RuntimeError('Python file changed: '+member.name)
                p.chmod(reference['files'][member.name]['mode'])
            elif member.issym():
                link=reference['symlinks'][member.name]
                if member.linkname!=link or Path(link).is_absolute():raise RuntimeError('Unexpected link')
                resolved=(p.parent/link).resolve()
                if target.resolve() not in resolved.parents:raise RuntimeError('Link escapes Python root')
                if p.is_symlink():
                    if str(p.readlink())!=link:raise RuntimeError('Existing Python link differs')
                elif p.exists():raise RuntimeError('Expected Python symlink')
                else:p.symlink_to(link)
            else:raise RuntimeError('Unsupported Python archive member')
    python=target/'bin/python3.11';venv=ROOT/'.venv'
    if not venv.exists():subprocess.run([str(python),'-m','venv','--without-pip',str(venv)],check=True)
    install_packages(venv)
    vp=venv/'bin/python'
    subprocess.run([str(vp),'-B',str(ROOT/'bootstrap.py')],check=True)
    subprocess.run([str(vp),'-B',str(ROOT/'verify_release.py')],check=True)

if __name__=='__main__':main()
