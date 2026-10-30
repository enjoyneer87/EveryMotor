"""Portable published-file integrity and saved-scalar arithmetic replay only.

Private original hashes are recorded provenance, not fetched or reverified here.
No model, checkpoint, native mesh, FEA, GPU or third-party package is required.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib
import json
import math
import statistics

ROOT=Path(__file__).resolve().parent
SEEDS=('17','29','43')
LIMITS={'seen9_Bvector':.5,'a1_Bvector':1.,'copper_Bvector':1.,'PA_ge70_case_mean_Bvector':1.,'pooled_torque':1.}
METRICS={'primary_all45_Bvector':'field/area_pooled/all45/all/vector_nrmse_percent',
         'seen9_Bvector':'field/area_pooled/seen9/all/vector_nrmse_percent',
         'a1_Bvector':'field/area_pooled/all45/a1/vector_nrmse_percent',
         'copper_Bvector':'field/area_pooled/all45/copper/vector_nrmse_percent',
         'PA_ge70_case_mean_Bvector':'field/PA_ge70_case_arithmetic/all45/all/vector_nrmse_percent',
         'pooled_torque':'torque/pooled/nrmse_percent'}


def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def same(a,b):
    assert type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)


def numeric_fingerprint(value):
    leaves=[]
    def walk(v,path=''):
        if isinstance(v,dict):
            for key in sorted(v):
                token=key.replace('~','~0').replace('/','~1')
                walk(v[key],path+'/'+token)
        elif isinstance(v,list):
            for index,child in enumerate(v):walk(child,path+'/'+str(index))
        elif type(v) in (int,float,bool) or v is None:
            leaves.append((path,type(v).__name__,v))
    walk(value)
    return sha(json.dumps(leaves,allow_nan=False,separators=(',',':')).encode())


def distribution(row):
    assert set(row['by_seed'])==set(SEEDS)
    values=[row['by_seed'][s] for s in SEEDS]
    for v in values:assert type(v) in (int,float) and math.isfinite(v)
    for key,value in (('mean',statistics.fmean(values)),('min',min(values)),('max',max(values)),('range',max(values)-min(values))):same(row[key],value)
    return values


class LocalLinks(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src') and value:self.links.append(value)


def main():
    manifest=read(ROOT/'source_manifest.json')
    assert manifest['status']=='COMPLETE_PUBLIC_REPORT_PACKAGE'
    assert manifest['scientific_scope']=='EXPOSED_INNERDEV_SAVED_ARITHMETIC_ONLY'
    assert manifest['private_originals_included'] is False
    assert manifest['numeric_fingerprint_format']=='json-pointer-type-value.v1'
    assert manifest['registered_training_updates']==265695 and manifest['registered_evaluation_graphs']==4320
    paths=set();indexed={};jsons=0
    for row in manifest['files']:
        q=(ROOT/row['path']).resolve();assert q.is_relative_to(ROOT.resolve())
        assert row['path'] not in paths;paths.add(row['path'])
        assert q not in indexed;indexed[q]=row
        assert q.suffix.lower() in ('.html','.png','.json','.py','.md')
        assert q.name.casefold()!='native_source_header_result.json','private raw header forbidden'
        data=q.read_bytes();assert len(data)==row['published_bytes'] and sha(data)==row['published_sha256'],q
        if row.get('numeric_fingerprint') is not None:
            assert q.suffix.lower()=='.json'
            assert numeric_fingerprint(json.loads(data))==row['numeric_fingerprint'];jsons+=1
    actual_files={q.resolve() for q in ROOT.rglob('*') if q.is_file()}
    assert actual_files==set(indexed)|{(ROOT/'source_manifest.json').resolve()},'actual files differ from strict manifest whitelist'
    summary_path=(ROOT/manifest['summary_file']).resolve()
    html_path=(ROOT/manifest['html_file']).resolve()
    assert summary_path in indexed and html_path in indexed
    assert indexed[summary_path].get('numeric_fingerprint') is not None
    summary=read(summary_path)
    assert summary['status']=='COMPLETE_SAVED_FINAL_WIDTH_DIAGNOSTICS'
    checked=0
    for metric,row in summary['metrics'].items():
        a,b=distribution(row['control']),distribution(row['candidate'])
        paired=distribution(row['paired_control_minus_candidate'])
        for c,d,delta in zip(a,b,paired):same(c-d,delta)
        checked+=1
    assert checked==summary['metric_count']==703
    verdict=summary['registered_judgment_replay']
    assert verdict['status']=='COMPLETE' and verdict['fixed_seeds']==[17,29,43]
    assert verdict['control_arm']=='saved_W64_phase45' and verdict['candidate_arm']=='W128_phase45_equal88565'
    for name,key in METRICS.items():
        for role in ('control','candidate'):
            distribution(verdict['metrics'][name][role])
            for seed in SEEDS:same(verdict['metrics'][name][role]['by_seed'][seed],summary['metrics'][key][role]['by_seed'][seed])
        row=verdict['metrics'][name]
        same(row['improvement_mean_pp'],row['control']['mean']-row['candidate']['mean'])
        same(row['worsening_mean_pp'],-row['improvement_mean_pp'])
        same(row['max_arm_range_pp'],max(row['control']['range'],row['candidate']['range']))
    primary=verdict['metrics']['primary_all45_Bvector'];delta=primary['improvement_mean_pp']
    primary_gates={'mean_improvement_at_least_1pp':delta>=1.,
                   'mean_improvement_strictly_gt_2max_range':delta>2*primary['max_arm_range_pp'],
                   'mean_improvement_strictly_gt_frozen_W64_phase45_bound':delta>2*primary['control']['range']}
    assert verdict['primary_gates']==primary_gates
    assert set(verdict['no_harm_gates'])==set(LIMITS)
    for name,limit in LIMITS.items():
        gate=verdict['no_harm_gates'][name];harm=verdict['metrics'][name]['worsening_mean_pp']
        same(gate['maximum_worsening_pp'],limit);same(gate['actual_worsening_pp'],harm)
        assert gate['pass']==(harm<=limit)
    passed=all(primary_gates.values()) and all(x['pass'] for x in verdict['no_harm_gates'].values())
    assert verdict['scientific_gates_passed']==passed
    assert verdict['verdict']==('IMPROVEMENT_CLAIM_WITHIN_INNERDEV' if passed else 'NO_IMPROVEMENT_CLAIM')
    parser=LocalLinks();parser.feed(html_path.read_text(encoding='utf-8'))
    local_count=0
    for value in parser.links:
        parsed=urlsplit(value)
        if parsed.scheme in ('https','http','mailto','data') or value.startswith('#'):continue
        assert not parsed.scheme,'unexpected external/local protocol'
        target=(ROOT/unquote(parsed.path)).resolve()
        assert target.is_relative_to(ROOT.resolve()) and target.is_file(),value
        local_count+=1
    print(json.dumps(dict(status='PASS_PUBLIC_FILE_INTEGRITY_AND_SAVED_ARITHMETIC_ONLY',
                         published_files_verified=len(paths),numeric_fingerprints_verified=jsons,
                         scalar_metric_distributions_replayed=checked,local_HTML_links_verified=local_count,
                         verdict=verdict['verdict'],new_training=0,new_inference=0,new_FEA=0,
                         private_original_hashes='recorded provenance; not independently fetched or verified by this portable command',
                         browser_rendering='not exercised; standalone PNG QA separate',
                         physical_BC_frame_pair='UNKNOWN; not established by this replay'),indent=2))


if __name__=='__main__':main()
