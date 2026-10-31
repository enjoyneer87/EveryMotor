"""Verify this public package and saved scalar arithmetic; no private/native proof."""
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href') and value:
                self.refs.append(value)


def Verify(root: Path) -> dict:
    manifest = json.loads((root / 'source_manifest.json').read_text(encoding='utf-8'))
    rows = manifest['files']
    expected = {row['path'] for row in rows} | {'source_manifest.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    assert len(rows) == len(expected) - 1 and expected == actual, 'Inventory mismatch'
    for row in rows:
        p = (root / row['path']).resolve()
        assert p.is_relative_to(root.resolve()) and not p.is_symlink()
        data = p.read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        if p.suffix in ('.json', '.html', '.md', '.py'):
            text = data.decode('utf-8')
            assert not re.search(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\[A-Za-z0-9]|(?i:license[_ -]?key|password\s*=|api[_ -]?key\s*=)', text), p.name
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'HOLD_NOT_RUN'
    assert not summary['native_admission'] and not summary['training_release']
    assert summary['new_training'] == summary['new_inference'] == summary['new_FEA'] == 0
    assert summary['numeric_compute_anchor'] is None
    assert summary['same_lab_angle_no_wrap_P0'] == 'FAIL_PRESERVED'
    assert summary['native_BC_sign_frame_pair'] == 'UNKNOWN'
    for row in summary['prior_completed_training']:
        value = row['primary_Bvector']
        assert set(value['by_seed']) == {'17', '29', '43'}
        samples = list(value['by_seed'].values())
        for key, expected_value in {'mean': sum(samples)/3, 'min': min(samples), 'max': max(samples), 'range': max(samples)-min(samples)}.items():
            assert math.isclose(value[key], expected_value, rel_tol=0, abs_tol=1e-12), key
    html = (root / 'index.html').read_text(encoding='utf-8')
    assert 'HOLD_NOT_RUN' in html and 'UNVERIFIED_SECURITY_POLICY_BLOCK' in html
    for row in summary['prior_completed_training']:
        for key in ('mean', 'min', 'max'):
            assert f"{row['primary_Bvector'][key]:.2f}" in html
    links = Links()
    links.feed(html)
    count = 0
    for ref in links.refs:
        url = urlsplit(ref)
        if url.scheme:
            assert url.scheme == 'https'
        elif url.path:
            target = (root / unquote(url.path)).resolve()
            assert target.is_relative_to(root.resolve()) and target.is_file(), ref
            count += 1
    qa = json.loads((root / 'QA.json').read_text(encoding='utf-8'))
    assert qa['status'] == 'PASS_STANDALONE_PNG_AND_STATIC_HTML'
    assert qa['browser_rendering'] == 'UNVERIFIED_SECURITY_POLICY_BLOCK'
    for name, sha in qa['reviewed_files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    return {'status': 'PASS_PUBLIC_INTEGRITY_SAVED_ARITHMETIC_AND_LOCAL_LINKS', 'files': len(expected),
            'saved_distributions': 3, 'local_links': count, 'new_model_execution': 0,
            'native_evidence_independently_retrieved': False, 'physical_admission': False,
            'browser_rendering': qa['browser_rendering']}


if __name__ == '__main__':
    print(json.dumps(Verify(Path(__file__).resolve().parent)))
