"""Auditor-only entry: reads paired reports and error bounds, no true costs."""
from pathlib import Path
import json,hashlib
import numpy as np
from paper_self_calibration_20261006 import closed_recover,self_calibration_certificate
root=Path(__file__).resolve().parent/'paper_breakthrough_20261006'
path=root/'external_paired_observations.json';records=json.loads(path.read_text());out=[]
for record in records:
    raw=np.array(record['paired_reported_aggregate_positions'])
    observed=(raw[:,1]-raw[:,0])/(2*record['input_amplitude'])
    estimate,sv,w=closed_recover(observed)
    certificate=self_calibration_certificate(observed,record['response_error_bound'])
    lo,hi=certificate['beta_interval'];cut=.99
    action='accept' if lo>=cut else 'fail_clock' if hi<cut else 'inconclusive'
    out.append(dict(label=record['replay_label'],parameter_order=['alpha/reference_cost','beta','cost_ratio_B','cost_ratio_C'],
        point=estimate.tolist(),certificate=certificate,clock_cut=cut,action=action,
        private_costs_or_targets_read=False,pencil_singular_values=sv.tolist()))
result=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),audits=out)
(root/'blind_observation_audit_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps([(r['label'],r['certificate']['beta_interval'],r['action']) for r in out]))
