"""Cost-calibration-aware interval inference and independent decision validation."""
from pathlib import Path
import itertools,json,math,hashlib,csv
from fractions import Fraction
import numpy as np
from independent_tracking_controller_20261006 import paired_impulse,task_loss
OUT=Path(__file__).resolve().parent/'paper_substantive_revision_20261006'
POOL=[(0.,.2),(.1,.3),(.7,1.),(.95,.5)]

def point(z,k):
    k=np.asarray(k);c=k*z[..., :,0]
    dc,db=c[...,2]-c[...,0],c[...,1]-c[...,0]
    a=dc[...,None]*(z[...,1,:]-z[...,0,:])-db[...,None]*(z[...,2,:]-z[...,0,:])
    b=dc[...,None]*(k[1]*z[...,1,:]-k[0]*z[...,0,:])-db[...,None]*(k[2]*z[...,2,:]-k[0]*z[...,0,:])
    det=a[...,0]*(b[...,2]-b[...,1])-a[...,1]*b[...,1]
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.stack([b[...,1]**2/det,a[...,0]*b[...,1]/det],-1)

def cost_gradient(z,k):
    c=np.asarray(k)*z[:,0];dc,db=c[2]-c[0],c[1]-c[0]
    v,w=z[1]-z[0],z[2]-z[0];vk=k[1]*z[1]-k[0]*z[0];wk=k[2]*z[2]-k[0]*z[0]
    a=dc*v-db*w;b=dc*vk-db*wk;det=a[0]*(b[2]-b[1])-a[1]*b[1]
    out=[]
    for j in range(3):
        h=np.eye(3)[j];dci=h*z[:,0];ddc,ddb=dci[2]-dci[0],dci[1]-dci[0]
        da=ddc*v-ddb*w
        d_b=ddc*vk+dc*(h[1]*z[1]-h[0]*z[0])-ddb*wk-db*(h[2]*z[2]-h[0]*z[0])
        dd=da[0]*(b[2]-b[1])+a[0]*(d_b[2]-d_b[1])-da[1]*b[1]-a[1]*d_b[1]
        out.append(((da[0]*b[1]+a[0]*d_b[1])*det-a[0]*b[1]*dd)/det**2)
    return np.array(out)*np.asarray(k)

def add(x,y):return np.nextafter(x[0]+y[0],-np.inf),np.nextafter(x[1]+y[1],np.inf)
def sub(x,y):return np.nextafter(x[0]-y[1],-np.inf),np.nextafter(x[1]-y[0],np.inf)
def mul(x,y):
    v=np.stack([x[0]*y[0],x[0]*y[1],x[1]*y[0],x[1]*y[1]])
    return np.nextafter(v.min(0),-np.inf),np.nextafter(v.max(0),np.inf)
def div(x,y):
    with np.errstate(divide='ignore',invalid='ignore'):
        inv=np.minimum(1/y[0],1/y[1]),np.maximum(1/y[0],1/y[1])
    return mul(x,(np.nextafter(inv[0],-np.inf),np.nextafter(inv[1],np.inf)))
def coord(x,n):return x[0][...,n],x[1][...,n]
def extra(x):return x[0][...,None],x[1][...,None]

def calibrated_box(z,t,ratios,relative_radius,subdivisions=16,extra_lags=True):
    """Outer interval hull over a COMPLETE cost-ratio box, not a point grid.

    Reference cost normalized to 1. r2,r3 intervals disjoint and ordered.
    Each cost tile includes its full continuous interior; elementary operations
    round outward. Singular tiles force a full-domain fallback. Necessary
    higher-order Euler residuals can only eliminate an impossible tile.
    Returns alpha/reference_cost and beta hulls, and empty-set indicator.
    """
    z=np.asarray(z);batch=z.ndim==3
    if not batch:z=z[None,:,:]
    radius=Fraction(str(relative_radius))
    lo=np.array([np.nextafter(float(Fraction(str(r))*(1-radius)),-np.inf) for r in ratios[1:]])
    hi=np.array([np.nextafter(float(Fraction(str(r))*(1+radius)),np.inf) for r in ratios[1:]])
    if relative_radius==0:lo=hi=np.asarray(ratios[1:])
    assert 1<lo[0] and hi[0]<lo[1]
    splits=1 if relative_radius==0 else subdivisions
    grids=[np.linspace(a,b,splits+1) for a,b in zip(lo,hi)]
    tiles=list(itertools.product(range(splits),repeat=2));C=len(tiles)
    # Every tile endpoint is outward enlarged; gaps cannot occur.
    k=[(np.ones((C,1)),np.ones((C,1)))]
    for j in range(2):
        k.append((np.nextafter(np.array([grids[j][a[j]] for a in tiles])[:,None],-np.inf),
                  np.nextafter(np.array([grids[j][a[j]+1] for a in tiles])[:,None],np.inf)))
    zi=[(np.nextafter(z[None,:,j,:]-t,-np.inf)+np.zeros((C,1,1)),
         np.nextafter(z[None,:,j,:]+t,np.inf)+np.zeros((C,1,1))) for j in range(3)]
    cs=[mul(k[j],coord(zi[j],0)) for j in range(3)]
    dc,db=extra(sub(cs[2],cs[0])),extra(sub(cs[1],cs[0]))
    aa=sub(mul(dc,sub(zi[1],zi[0])),mul(db,sub(zi[2],zi[0])))
    bb=sub(mul(dc,sub(mul(extra(k[1]),zi[1]),mul(extra(k[0]),zi[0]))),
           mul(db,sub(mul(extra(k[2]),zi[2]),mul(extra(k[0]),zi[0]))))
    a0,a1,b1=coord(aa,0),coord(aa,1),coord(bb,1)
    dd=sub(mul(a0,sub(coord(bb,2),b1)),mul(a1,b1))
    defined=(dd[0]>0)&(b1[1]<0)
    impossible=(dd[1]<=0)|(b1[0]>=0)
    sq=(np.nextafter(np.minimum(b1[0]**2,b1[1]**2),-np.inf),np.nextafter(np.maximum(b1[0]**2,b1[1]**2),np.inf))
    ai=div(sq,dd);bi=div(mul(a0,b1),dd)
    # Invalid floating arithmetic never establishes an exclusion.
    undefined_numeric=~(np.isfinite(ai[0])&np.isfinite(ai[1])&np.isfinite(bi[0])&np.isfinite(bi[1])&
        np.isfinite(dd[0])&np.isfinite(dd[1])&np.isfinite(b1[0])&np.isfinite(b1[1]))
    defined &= ~undefined_numeric
    impossible &= ~undefined_numeric
    al=np.where(defined,np.maximum(0.,ai[0]),0.);au=np.where(defined,ai[1],np.inf)
    bl=np.where(defined,np.maximum(0.,bi[0]),0.);bu=np.where(defined,np.minimum(1.,bi[1]),1.)
    impossible|=defined&((au<=0)|(bu<0)|(bl>=1)|(bl>bu))
    if extra_lags and z.shape[-1]>3:
        for n in range(2,z.shape[-1]-1):
            un=sub(coord(bb,n),coord(bb,n-1));unext=sub(coord(bb,n+1),coord(bb,n))
            rr=add(sub(mul((al,au),coord(aa,n)),mul((bl,bu),unext)),un)
            # Do not use inf*0 fallback arithmetic to prune singular tiles.
            impossible|=defined&((rr[0]>0)|(rr[1]<0))
    good=~impossible
    al=np.min(np.where(good,al,np.inf),axis=0);au=np.max(np.where(good,au,-np.inf),axis=0)
    bl=np.min(np.where(good,bl,np.inf),axis=0);bu=np.max(np.where(good,bu,-np.inf),axis=0)
    empty=~np.any(good,axis=0)
    if batch:return al,au,bl,bu,empty
    return tuple(x[0] for x in (al,au,bl,bu,empty))

def threshold(sigma,m,coords,delta=.05):
    value=sigma*math.sqrt(2*math.log(2*coords/delta)/m)*(1+1e-12)
    q=Fraction(m)*Fraction(value)**2/(2*Fraction(sigma)**2)
    partial=term=Fraction(1)
    for j in range(1,180):term=term*q/j;partial+=term
    assert partial>Fraction(2*coords)/Fraction(str(delta))
    return value

def overidentified(z,k,lags):
    mats=[];rhs=[]
    for j in range(2,len(k)):
        v=z[..., [0,1,j],:lags];kk=np.asarray(k)[[0,1,j]]
        c=v[..., :,0]*kk;dc,db=c[...,2]-c[...,0],c[...,1]-c[...,0]
        a=dc[...,None]*(v[...,1,:]-v[...,0,:])-db[...,None]*(v[...,2,:]-v[...,0,:])
        b=dc[...,None]*(kk[1]*v[...,1,:]-kk[0]*v[...,0,:])-db[...,None]*(kk[2]*v[...,2,:]-kk[0]*v[...,0,:])
        u=b.copy();u[...,1:]-=b[...,:-1]
        mats.append(np.stack([a[...,:-1],-u[...,1:]],-1));rhs.append(-u[...,:-1])
    X=np.concatenate(mats,-2);y=np.concatenate(rhs,-1)
    gram=np.swapaxes(X,-1,-2)@X;xy=np.einsum('...ni,...n->...i',X,y)
    return np.linalg.solve(gram,xy[...,None])[...,0]

def decision(lo,hi,empty,cut=.6):
    return np.where(empty,'model_incompatible',np.where(lo>=cut,'accept',np.where(hi<cut,'fail_clock','inconclusive')))

def noise(rng,reps,M,L,sigma,m,kind='gaussian'):
    if kind=='gaussian':
        return sigma/math.sqrt(m)*(np.sqrt(.5)*rng.normal(size=(reps,1,1))+np.sqrt(.5)*rng.normal(size=(reps,M,L)))
    # iid Rademacher episode means; an exact bounded/subGaussian law.
    return sigma*(2*rng.binomial(m,.5,size=(reps,M,L))/m-1)

def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if math.isfinite(float(x)) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    return x

def main():
    OUT.mkdir(exist_ok=True);rng=np.random.default_rng(2026100603)
    sensitivity=[]
    for label,k in [('ordinary',[1.,2.,4.]),('wide',[.5,2.,8.]),('close',[1.,1.1,1.2])]:
        z,meta=paired_impulse(2.5,2/3,k,POOL,[.5,.35,.15],8)
        g=cost_gradient(z,k)
        assert abs(g.sum())<1e-6
        for j in range(3):
            kc=np.array(k,dtype=complex);kc[j]+=1e-25j
            assert abs(point(z,kc)[1].imag/1e-25*k[j]-g[j])<1e-5*max(1.,abs(g[j]))
        for error in [-.01,-.001,0.,.001,.01]:
            nominal=k.copy();nominal[1]*=1+error
            est=point(z,nominal)
            sensitivity.append(dict(cost_design=label,costs=k,middle_relative_error=error,alpha=est[0],beta=est[1],relative_cost_gradient=g.tolist(),local_ratio_amplification=float(abs(g[1])+abs(g[2])),solver=meta))
    calibrated=[]
    z,_=paired_impulse(2.5,2/3,[1.,2.,4.],POOL,[.5,.35,.15],8)
    for radius in [0.,.0001,.001,.01]:
        for m in [100,10000]:
            t=threshold(1e-4,m,24)
            ci=calibrated_box(z,t,[1.,2.,4.],radius,32)
            assert not ci[-1] and ci[0]<=2.5<=ci[1] and ci[2]<=2/3<=ci[3]
            calibrated.append(dict(radius=radius,m=m,beta_lower=ci[2],beta_upper=ci[3],beta_width=ci[3]-ci[2],response_half_width=t))
    # One-sided nominal miscalibration would wrongly fail a .6 acceptance clock.
    nominal=[1.,2.02,4.]
    miscalibration_ci=calibrated_box(z,0.,nominal,.011,64)
    assert miscalibration_ci[2]<=2/3<=miscalibration_ci[3]
    witness=dict(nominal_costs=nominal,true_costs=[1,2,4],naive_beta=point(z,nominal)[1],
                 robust_beta_interval=list(miscalibration_ci[2:4]),
                 robust_decision=decision(*miscalibration_ci[2:],cut=.6).item())
    # Finite validation with errors at calibration-box edges and inside the
    # simultaneous response box. Inclusion, not MC frequency, is the invariant.
    checks=0
    for beta in [0.,.3,2/3,.98]:
        for filt in [[.5,.35,.15],[-.5,1.2,.3]]:
            for cost_edge in itertools.product([-1,1],repeat=2):
                costs=[1.,2*(1+.001*cost_edge[0]),4*(1+.001*cost_edge[1])]
                exact,_=paired_impulse(2.5,beta,costs,POOL,filt,4)
                t=1e-7
                perturbed=exact[None,:,:]+rng.uniform(-t,t,size=(40,3,4))
                ci=calibrated_box(perturbed,t,[1.,2.,4.],.001,16)
                assert np.all(~ci[-1]) and np.all(ci[0]<=2.5) and np.all(ci[1]>=2.5)
                assert np.all(ci[2]<=beta) and np.all(ci[3]>=beta)
                checks+=40
    decisions=[];raw_logs={};pilot_audits={};reps=500
    for label,beta,alpha,pool,filt,cal,sigma,m,kind in [
      ('correct_clock',.8,2.5,POOL,[.5,.35,.15],.0001,1e-4,1000,'gaussian'),
      ('wrong_clock',.45,2.5,POOL,[.5,.35,.15],.0001,1e-4,1000,'gaussian'),
      ('near_threshold',.6,2.5,POOL,[.5,.35,.15],.0001,1e-4,1000,'gaussian'),
      ('imprecise_costs',.8,2.5,POOL,[.5,.35,.15],.01,1e-4,1000,'gaussian'),
      ('weak_report',.8,2.5,POOL,[.01,.09,.9],.0001,1e-4,1000,'gaussian'),
      ('signed_dense_pool',.8,.5,[(i/64,1/64) for i in range(64)],[-.5,1.2,.3],.0001,1e-5,1000,'rademacher'),
      ('large_penalty',.8,10.,[(.05,.2),(.9,.8)],[.5,.35,.15],.0001,1e-4,1000,'gaussian')]:
        costs=[.5,2.,8.];ratios=[1.,4.,16.];L=8
        true,meta=paired_impulse(alpha,beta,costs,pool,filt,L)
        obs=true[None,:,:]+noise(rng,reps,3,L,sigma,m,kind)
        t=threshold(sigma,m,3*L)
        parts=[calibrated_box(obs[j:j+50],t,ratios,cal,16) for j in range(0,reps,50)]
        ci=[np.concatenate([p[j] for p in parts]) for j in range(5)]
        action=decision(*ci[2:]);joint=np.max(abs(obs-true),axis=(1,2))<=t
        cover=(ci[0]<=alpha/.5)&(ci[1]>=alpha/.5)&(ci[2]<=beta)&(ci[3]>=beta)&~ci[-1]
        assert np.all(cover[joint])
        wrong=(action=='accept')&(beta<.6) | (action=='fail_clock')&(beta>=.6) | (action=='model_incompatible')
        assert not np.any(wrong[joint])
        decisions.append(dict(case=label,beta=beta,alpha=alpha,noise_kind=kind,sigma=sigma,m=m,repetitions=reps,calibration_radius=cal,
          beta_median_width=float(np.median(ci[3]-ci[2])),coverage=float(cover.mean()),
          actions=dict(zip(*np.unique(action,return_counts=True))),wrong_decision_rate=float(wrong.mean()),
          coefficient_event_rate=float(joint.mean()),snr_early=float(np.min(abs(true[:,0]))/sigma),solver=meta))
        if label in ['correct_clock','wrong_clock']:
            # Explicit episode records for one independent paired replay pilot.
            raw=true[None,:,:]+noise(rng,m,3,L,sigma,1,kind)
            raw_logs[label]=dict(observations=raw.tolist(),calibrated_cost_ratios=ratios,calibration_radius=cal,sigma=sigma,
                                 input_amplitude=1.,lags=L,repetitions=m,observable='paired signed aggregate positions')
            pilot_ci=calibrated_box(raw.mean(0),t,ratios,cal,32)
            pilot_audits[label]=dict(beta_interval=list(pilot_ci[2:4]),alpha_over_reference_interval=list(pilot_ci[:2]),
                action=decision(*pilot_ci[2:]).item(),response_half_width=t,repetitions=m,
                nominal_cost_ratios=ratios,ratio_radius=cal,noise_upper_bound=sigma,acceptance_cut=.6)
            assert pilot_audits[label]['action']==('accept' if label=='correct_clock' else 'fail_clock')
    efficiency=[]
    for beta in [.3,2/3,.98]:
        for alpha in [.5,2.5,10.]:
            costs=[.5,1.,2.,4.,8.];z,_=paired_impulse(alpha,beta,costs,POOL,[.5,.35,.15],8)
            obs=z[None,:,:]+noise(rng,2000,5,8,1e-4,1000)
            short=point(obs[:,[0,2,4],:],np.array(costs)[[0,2,4]])
            long=overidentified(obs[:,[0,2,4],:],[.5,2.,8.],8)
            multi=overidentified(obs,costs,8)
            for name,estimate in [('three_cost_three_lag',short),('three_cost_eight_lag',long),('five_cost_eight_lag',multi)]:
                efficiency.append(dict(alpha=alpha,beta=beta,method=name,reps=2000,
                    beta_median_absolute_error=float(np.median(abs(estimate[:,1]-beta))),
                    beta_rmse=float(np.sqrt(np.mean((estimate[:,1]-beta)**2))),
                    invalid_rate=float(np.mean((estimate[:,0]<=0)|(estimate[:,1]<0)|(estimate[:,1]>=1)))))
    loss={str(b):task_loss(2.5,b,.8,2.,POOL) for b in [.45,.6,.8]}
    assert loss['0.8']<loss['0.45']
    result=clean(dict(seed=2026100603,independent_provider='discounted 2-state Bellman iteration; paired nonzero-state replay',
       sensitivity=sensitivity,calibrated_intervals=calibrated,miscalibration_witness=witness,
       deterministic_response_and_cost_edge_checks=checks,decision_cases=decisions,pilot_audits=pilot_audits,
       efficiency_comparison=efficiency,mission_loss=loss,
       mission_clock=.8,mission_cost=2.,mission_innovation_variance=.04,
       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       provider_sha256=hashlib.sha256((Path(__file__).parent/'independent_tracking_controller_20261006.py').read_bytes()).hexdigest(),
       limitations=['Controlled independent implementation, not observed institution or human behavior.',
                   'Ratio calibration intervals and noise upper bounds are supplied externally, not estimated from the same episodes.',
                   'Long-prefix OLS is a point-estimate benchmark, not a certified interval or efficient estimator.',
                   'MC frequencies are not guaranteed powers; deterministic inclusion is tested only on finitely many configurations.']))
    (OUT/'calibration_and_decision_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'paired_replay_observation_logs.json').write_text(json.dumps(clean(raw_logs),ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(clean(dict(checks=checks,witness=witness,calibration=calibrated,decisions=decisions,mission_loss=loss)),ensure_ascii=False))

if __name__=='__main__':main()
