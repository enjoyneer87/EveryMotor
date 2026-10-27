"""Refresh only artifact hashes; never regenerate scientific results."""
from pathlib import Path
from field_reader import write_json,sha

def refresh():
    out=Path(__file__).resolve().parent
    for f in out.rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts and f.suffix in {'.py','.md','.json','.csv','.txt'}:
            text=f.read_text(encoding='utf-8-sig');f.write_text(text,encoding='utf-8',newline='\n')
    files=[dict(path=f.relative_to(out).as_posix(),sha256=sha(f),bytes=f.stat().st_size)
        for f in sorted(out.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='artifact_manifest.json']
    write_json(out/'artifact_manifest.json',dict(schema='return-artifact-hashes-v1',files=files,total_bytes=sum(f['bytes'] for f in files),
        exclude='artifact_manifest.json self-reference and ignored __pycache__',source_originals_modified=False))
    print(dict(files=len(files),bytes=sum(x['bytes'] for x in files)))

if __name__=='__main__':refresh()
