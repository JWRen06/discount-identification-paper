"""Exclude previously used WTI demonstration dates from evaluation scoring.

This is a retrospective overlap sensitivity check, not a newly unseen sample.
Frozen training, module, controller state recursion, and all clocks are retained.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from paper_real_path_holdout_20261006 import data,run,metrics,CUT,BETAS
from paper_stability_revision_20261006 import OUT,ROOT

def main():
    old=json.loads((ROOT/'later_holdout_results.json').read_text());selected=old['selected_common_module'];result=[]
    logs=json.loads((ROOT/'external_paired_observations.json').read_text())
    # Source log dates are used directly, not a hand-selected calendar cut.
    dates=logs[0]['baseline_prices_dates']
    for name,rows in data().items():
        cut=sum(day<=CUT for day,value in rows)-1
        evaluation_dates=[day.isoformat() for day,p in rows[cut+1:]]
        mask=np.array([not (name=='WTI_EIA' and day in dates) for day in evaluation_dates])
        assert sum(~mask)==(32 if name=='WTI_EIA' else 0)
        cases={}
        for beta in BETAS:
            loss,meta,forecast,u=run(rows,cut,beta,selected);cases[str(beta)]=metrics(loss[cut:][mask])
        result.append(dict(series=name,excluded=int(sum(~mask)),evaluation_start=next(day for day,keep in zip(evaluation_dates,mask) if keep),
                           evaluation_end=evaluation_dates[-1],cases=cases,improvement_vs_045=1-cases['0.9999']['discounted_mean_loss']/cases['0.45']['discounted_mean_loss'],
                           improvement_vs_080=1-cases['0.9999']['discounted_mean_loss']/cases['0.8']['discounted_mean_loss'],
                           improvement_vs_099=1-cases['0.9999']['discounted_mean_loss']/cases['0.99']['discounted_mean_loss']))
    OUT.mkdir(exist_ok=True)
    value=dict(historical_overlap_dates=dates,selected_module=selected,results=result,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               interpretation='Retrospective overlap exclusion; previously evaluated remaining dates are not new unseen validation. Initial policy states keep historical recursion; exclusion changes scoring only.')
    (OUT/'overlap_excluded_results.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([(r['series'],r['excluded'],r['improvement_vs_045'],r['improvement_vs_080'],r['improvement_vs_099']) for r in result]))

if __name__=='__main__':main()
