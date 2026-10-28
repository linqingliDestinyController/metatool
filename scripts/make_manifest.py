"""Regenerate the release file manifest after an intentional reviewed code change."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest={}
for p in sorted(root.rglob('*')):
    if not p.is_file() or any(n in {'.git','.venv','__pycache__'} for n in p.parts):continue
    if p.name=='MANIFEST.json' or p.suffix in {'.pyc','.log'}:continue
    manifest[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'Bound {len(manifest)} release files')
