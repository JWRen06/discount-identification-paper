from pathlib import Path
import json,hashlib,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
base=Path(__file__).resolve().parent
names=['switch_common_filter_positive_worlds','switch_three_cost_three_lag_positive_pool','switch_three_cost_two_lag_positive_base','switch_three_cost_finite_noise_certificate','switch_three_cost_finite_filter_signal_floor','switch_three_cost_heterogeneous_risk_rejection']
rows=[]
with tempfile.TemporaryDirectory(prefix='p09_paper_review_') as temp:
    dst=Path(temp)
    for name in names+['switch_three_cost_common_filter_identification']:
        shutil.copy2(base/(name+'.py'),dst/(name+'.py'))
    for name in names:
        proc=subprocess.run([sys.executable,str(dst/(name+'.py'))],capture_output=True,text=True,timeout=90)
        assert proc.returncode==0,(name,proc.stderr)
        actual=json.loads((dst/(name+'_results.json')).read_text(encoding='utf-8'))
        saved=json.loads((base/(name+'_results.json')).read_text(encoding='utf-8'))
        assert actual==saved,name+' differs from archived result'
        rows.append({'entry':name,'label':actual['label'],'exact_archived_result_match':True,'source_sha256':hashlib.sha256((base/(name+'.py')).read_bytes()).hexdigest()})
    obs=base/'market_observation'
    for name in ['mtum_two_snapshot_fields_20261006.json','mtum_two_snapshot_observation_20261006.py']:
        shutil.copy2(obs/name,dst/name)
    proc=subprocess.run([sys.executable,str(dst/'mtum_two_snapshot_observation_20261006.py')],capture_output=True,text=True,timeout=90)
    assert proc.returncode==0,proc.stderr
    actual=json.loads((dst/'mtum_two_snapshot_results_20261006.json').read_text(encoding='utf-8'))
    saved=json.loads((obs/'mtum_two_snapshot_results_20261006.json').read_text(encoding='utf-8'))
    assert actual==saved
    rows.append({'entry':'mtum_two_snapshot_observation_20261006','label':'M01','exact_archived_result_match':True,'source_sha256':hashlib.sha256((obs/'mtum_two_snapshot_observation_20261006.py').read_bytes()).hexdigest()})
out={'review':'manuscript reproduction, not new research experiments','created_at_utc':datetime.now(timezone.utc).isoformat(),'isolated_temporary_directory':True,'archived_outputs_overwritten':False,'passed':len(rows),'checks':rows,'limitations':'Reproduces numeric certificates; not a substitute for theorem proofs or independent market-data extraction.'}
reports=base/'paper_review_reports';reports.mkdir(exist_ok=True)
p=reports/('reproduction_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
p.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'passed':len(rows),'report':str(p)},ensure_ascii=False))
