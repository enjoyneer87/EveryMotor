"""Recompute small public artifact hashes after final documentation changes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
files=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and p.name not in ['ARTIFACT_MANIFEST.json','PACKAGE_VERIFICATION.json']:
        files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(dict(files=files,total_bytes=sum(x['bytes'] for x in files),exclusions=['manifest self','PACKAGE_VERIFICATION.json self-referential receipt','Python cache']),indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=sum(x['bytes'] for x in files))))
