"""Cost-free audit reading labels and observations only."""
import json,hashlib
import numpy as np
from paper_stability_revision_20261006 import OUT,certificate
path=OUT/'noisy_paired_observations.json';rows=[]
for r in json.loads(path.read_text()):
    raw=np.asarray(r['paired_reports']);z=(raw[:,1]-raw[:,0])/(2*r['input_amplitude'])
    c=certificate(z,r['response_error_bound']);lo,hi=c['beta_interval']
    rows.append(dict(label=r['label'],certificate=c,decision='accept' if lo>=.99 else 'fail' if hi<.99 else 'undetermined'))
(OUT/'noisy_observer_audit.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),results=rows),indent=2)+'\n',encoding='utf-8')
print(json.dumps([(r['label'],r['decision']) for r in rows]))
