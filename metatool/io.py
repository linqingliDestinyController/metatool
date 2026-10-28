"""Small, explicit file/binding utilities. Never mutate the parent run."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def atomic(path, obj):
    path = Path(path)
    if path.exists():
        if read(path) != obj:
            raise FileExistsError(f'Different committed evidence: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def source_identity():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts and
            (p.suffix in {'.py', '.sh'} or p.parent.name == 'configs')}


def model_identity(path):
    """Hash all base weights ONCE per invocation, not merely size/mtime."""
    path = Path(path).resolve()
    files = sorted(path.glob('*.safetensors'))
    if not files:
        raise ValueError('Expected complete local safetensors')
    names = files + sorted(path.glob('*.json')) + list(path.glob('merges.txt'))
    return {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in names}


