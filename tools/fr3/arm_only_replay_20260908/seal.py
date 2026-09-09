"""Seal this independent bundle after offline tests, without changing old code."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
assert not (root/'manifest.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()
       and not set(p.relative_to(root).parts)&{'logs','build','__pycache__'}
       and p.name not in ('manifest.json','session.lock')}
with (root/'manifest.json').open('x') as f:
    json.dump(dict(files=files,libfranka_sha256=sha(Path('/usr/local/lib/libfranka.so.0.15.0')),
                   physical_validation_completed=False),f,indent=2)
