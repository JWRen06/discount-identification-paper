"""Predefined module selection and later EIA risk-path holdout evaluation."""
from pathlib import Path
import csv,json,datetime,hashlib
import numpy as np
from paper_external_rule_replay_20261006 import make_agent,step,NODES,WEIGHTS,ROOT,EXT

METHODS=['zero','fixed_ar1','rolling_ar1_250']
CUT=datetime.datetime(2022,5,23)
BETAS=[.45,.6,.8,.95,.99,.9999]
ALPHA=.9999/15

def data():
    manifest=json.loads((EXT/'eia_source_manifest.json').read_text())
    out={}
    for item in manifest['sources']:
        path=EXT/(item['name']+'_full.csv')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['full_csv_sha256']
        rows=[(datetime.datetime.fromisoformat(r['date']),float(r['price'])) for r in csv.DictReader(path.open())]
        out[item['name']]=rows
    return out

def ar_coefficients(u):
    X=np.stack([np.ones(len(u)-1),u[:-1]],1);a,r=np.linalg.lstsq(X,u[1:],rcond=None)[0]
    return float(a),float(np.clip(r,-.95,.95))

def run(rows,training_end,beta,method):
    changes=np.diff([p for d,p in rows]);center=float(np.mean(changes[:training_end]));sigma=float(np.std(changes[:training_end],ddof=1))
    u=(changes-center)/sigma;c,r=ar_coefficients(u[:training_end]);parameters=[]
    for i in range(len(u)):
        if method=='zero':parameters.append((0.,0.))
        elif method=='fixed_ar1':parameters.append((c,r))
        else:parameters.append(ar_coefficients(u[max(0,i-249):i+1]) if i>=249 else (c,r))
    parameters=np.array(parameters);loss=np.zeros(len(u));forecast=np.zeros(len(u))
    for j,s in enumerate(NODES):
        agent=make_agent(ALPHA,beta,1.,s);E=step(agent,1.,0.);C=step(agent,0.,1.);bE=beta*E
        p=d=0.
        for i,signal in enumerate(u):
            d=s*d+(1-s)*signal;intercept,rr=parameters[i];mean=intercept/(1-rr)
            predicted_series=mean*bE/(1-bE)+(signal-mean)*bE*rr/(1-bE*rr)
            pn=step(agent,p,d)+C*(1-s)*predicted_series
            loss[i]+=WEIGHTS[j]*(.5*ALPHA*(pn-d)**2+.5*(pn-p)**2);p=pn
            forecast[i]=intercept+rr*signal
    return loss,dict(center=center,scale=sigma,fixed_ar_intercept=c,fixed_ar_coefficient=r),forecast,u

def metrics(loss):
    w=.9999**np.arange(len(loss))
    return dict(observations=len(loss),mean_loss=float(np.mean(loss)),discounted_mean_loss=float(np.dot(w,loss)/w.sum()))

def main():
    prices=data();selection={m:[] for m in METHODS};development=[]
    for name,rows in prices.items():
        cut=sum(day<=CUT for day,value in rows)-1;train=int(.7*cut)
        results={}
        for method in METHODS:
            loss,meta,f,u=run(rows[:cut+1],train,.9999,method);results[method]=metrics(loss[train:cut])
        for method in METHODS:selection[method].append(results[method]['discounted_mean_loss']/results['zero']['discounted_mean_loss'])
        development.append(dict(series=name,training_observations=train,validation_observations=cut-train,results=results))
    scores={method:float(np.mean(values)) for method,values in selection.items()}
    selected=min(METHODS,key=lambda m:scores[m]);holdout=[]
    for name,rows in prices.items():
        cut=sum(day<=CUT for day,value in rows)-1;cases={};module_cases={};years={}
        for beta in BETAS:
            loss,meta,forecast,u=run(rows,cut,beta,selected);cases[str(beta)]=metrics(loss[cut:])
            days=np.array([day.year for day,p in rows[cut+1:]])
            years[str(beta)]={str(y):metrics(loss[cut:][days==y]) for y in sorted(set(days))}
        for method in METHODS:
            loss,_meta,_f,_u=run(rows,cut,.9999,method);module_cases[method]=metrics(loss[cut:])
        holdout.append(dict(series=name,test_start=rows[cut+1][0].isoformat(),test_end=rows[-1][0].isoformat(),
            frozen_training=meta,controller_cases=cases,year_cases=years,forecast_module_cases=module_cases,
            improvement_vs_045=1-cases['0.9999']['discounted_mean_loss']/cases['0.45']['discounted_mean_loss'],
            improvement_vs_080=1-cases['0.9999']['discounted_mean_loss']/cases['0.8']['discounted_mean_loss'],
            improvement_vs_099=1-cases['0.9999']['discounted_mean_loss']/cases['0.99']['discounted_mean_loss'],
            module_improvement_vs_zero=1-module_cases[selected]['discounted_mean_loss']/module_cases['zero']['discounted_mean_loss']))
    result=dict(protocol_sha256=hashlib.sha256((ROOT/'后续留出检验方案.md').read_bytes()).hexdigest(),
        module_candidates=METHODS,selection_scores=scores,selected_common_module=selected,development=development,
        later_holdout=holdout,alpha=ALPHA,cost=1.,task_beta=.9999,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Local preanalysis plan, not external registration.','Historical prices are real; exposures and actions are counterfactual.',
                    'No guarantee that estimated forecasts equal true conditional expectations.',
                    'Software objective costs do not measure actual transaction costs.'])
    (ROOT/'later_holdout_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(scores=scores,selected=selected,holdout=[(h['series'],h['improvement_vs_045'],h['improvement_vs_080'],h['improvement_vs_099'],h['module_improvement_vs_zero']) for h in holdout])))

if __name__=='__main__':main()
