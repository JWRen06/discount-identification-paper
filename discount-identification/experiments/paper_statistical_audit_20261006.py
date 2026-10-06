"""Fixed-environment impulse experiment; Gaussian sample-mean simulation.

No market data, no bootstrap, and no truncation of infinite impulse responses:
the first four coefficients are evaluated directly from the policy kernel.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import NormalDist
from fractions import Fraction
from functools import lru_cache
import numpy as np

OUT = Path(__file__).resolve().parent / 'paper_upgrade_results_20261006'
SEED = 20261006
REPS = 1000
DELTA = .05
ALPHA = 2.5
POOL = [(.0, .2), (.1, .3), (.7, 1.0), (.95, .5)]

@lru_cache(None)
def tail_threshold(sigma,m):
    """Certify exp(m*t^2/(2*sigma^2)) > 24/delta with rational arithmetic.

    A finite Taylor sum is a lower bound for exp at a positive argument.
    This certifies that the chosen floating threshold is above the analytic
    tail budget without trusting floating logarithm accuracy.
    """
    target=Fraction(24)/Fraction(str(DELTA))
    t=sigma*math.sqrt(2*math.log(float(target))/m)*(1+1e-12)
    for _ in range(4):
        q=Fraction(m)*Fraction(t)**2/(2*Fraction(sigma)**2)
        term=Fraction(1);partial=term
        for j in range(1,161):
            term=term*q/j;partial+=term
            if partial>target:return t
        t=np.nextafter(t*(1+1e-12),np.inf)
    raise AssertionError('Could not certify the tail threshold')

def response(alpha, beta, costs, pool, filt, n=4):
    rows = []
    for k in costs:
        r = 1 + beta + alpha / k
        e = 2 / (r + math.sqrt(r*r - 4*beta))
        f = np.array([sum(w*(1-e)*(1-beta*e)*(1-s)/(1-beta*e*s)
                              * sum(e**j*s**(h-j) for j in range(h+1))
                              for s, w in pool) for h in range(n)])
        rows.append(np.convolve(f, filt)[:n])
    return np.array(rows)

def terms(z, costs):
    k = np.array(costs)
    c = z[..., :, 0]*k
    dc, db = c[..., 2]-c[..., 0], c[..., 1]-c[..., 0]
    v, w = z[..., 1, :]-z[..., 0, :], z[..., 2, :]-z[..., 0, :]
    vk = k[1]*z[..., 1, :]-k[0]*z[..., 0, :]
    wk = k[2]*z[..., 2, :]-k[0]*z[..., 0, :]
    a = dc[..., None]*v-db[..., None]*w
    b = dc[..., None]*vk-db[..., None]*wk
    return a, b, dc, db, v, w, vk, wk

def estimate(z, costs, eps):
    a,b,dc,db,v,w,vk,wk = terms(z, costs)
    u = b.copy(); u[..., 1:] -= b[..., :-1]
    determinant = a[..., 0]*u[..., 2]-a[..., 1]*b[..., 1]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ah = b[..., 1]**2/determinant
        bh = a[..., 0]*b[..., 1]/determinant
        # Inverse infinity norm of [[A0,-B1],[A1,-u2]].
        q = np.maximum(np.abs(u[..., 2])+np.abs(b[..., 1]),
                       np.abs(a[..., 1])+np.abs(a[..., 0]))/np.abs(determinant)
        km = max(costs)
        ae = 2*eps*(np.abs(dc)+np.abs(db))+2*km*eps*(np.abs(v).sum(-1)+np.abs(w).sum(-1))+8*km*eps**2
        be = 2*km*eps*(np.abs(dc)+np.abs(db))+2*km*eps*(np.abs(vk).sum(-1)+np.abs(wk).sum(-1))+8*km**2*eps**2
        eta, rho = ae+2*be, 2*be
        usable = np.isfinite(q) & (q*eta < 1)
        rad = np.where(usable, q*(rho+eta*np.maximum(np.abs(ah),np.abs(bh)))/(1-q*eta), np.inf)
        residual = ah*a[..., 2]-bh*u[..., 3]+u[..., 2]
        cap = rad*(np.abs(a[..., 2])+np.abs(u[..., 3]))+(np.abs(ah)+rad)*ae+(np.abs(bh)+rad)*2*be+2*be
    valid = np.isfinite(ah) & np.isfinite(bh) & (ah > 0) & (bh >= 0) & (bh < 1)
    reject = usable & (np.abs(residual) > cap)
    return ah,bh,rad,usable,valid,reject

def jacobian(z, costs):
    """Analytic derivative with respect to the twelve observed coefficients."""
    a,b,dc,db,v,w,vk,wk=terms(z,costs)
    det=a[...,0]*(b[...,2]-b[...,1])-a[...,1]*b[...,1]
    gradients=[]
    for environment in range(3):
        for lag in range(4):
            direction=np.zeros((3,4));direction[environment,lag]=1
            d_c=direction[:,0]*np.array(costs)
            d_dc,d_db=d_c[2]-d_c[0],d_c[1]-d_c[0]
            d_v,d_w=direction[1]-direction[0],direction[2]-direction[0]
            d_vk=costs[1]*direction[1]-costs[0]*direction[0]
            d_wk=costs[2]*direction[2]-costs[0]*direction[0]
            da=d_dc*v+dc[...,None]*d_v-d_db*w-db[...,None]*d_w
            d_b=d_dc*vk+dc[...,None]*d_vk-d_db*wk-db[...,None]*d_wk
            d_det=da[...,0]*(b[...,2]-b[...,1])+a[...,0]*(d_b[...,2]-d_b[...,1])-da[...,1]*b[...,1]-a[...,1]*d_b[...,1]
            with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
                ga=(2*b[...,1]*d_b[...,1]*det-b[...,1]**2*d_det)/det**2
                gb=((da[...,0]*b[...,1]+a[...,0]*d_b[...,1])*det-a[...,0]*b[...,1]*d_det)/det**2
            gradients.append(np.stack([ga,gb],axis=-1))
    return np.stack(gradients,axis=-1)

def interval_confidence(z,costs,t):
    """Outward interval evaluation of the exact recovery map.

    Each elementary floating operation is rounded outwards. Dependencies are
    retained only as shared input intervals, so enclosures remain conservative.
    """
    def add(x,y):return np.nextafter(x[0]+y[0],-np.inf),np.nextafter(x[1]+y[1],np.inf)
    def sub(x,y):return np.nextafter(x[0]-y[1],-np.inf),np.nextafter(x[1]-y[0],np.inf)
    def mul(x,y):
        vals=np.stack([x[0]*y[0],x[0]*y[1],x[1]*y[0],x[1]*y[1]])
        return np.nextafter(vals.min(0),-np.inf),np.nextafter(vals.max(0),np.inf)
    def div(x,y):
        with np.errstate(divide='ignore',invalid='ignore'):
            inv=np.minimum(1/y[0],1/y[1]),np.maximum(1/y[0],1/y[1])
        return mul(x,(np.nextafter(inv[0],-np.inf),np.nextafter(inv[1],np.inf)))
    def scale(x,k):return mul(x,(np.array(k),np.array(k)))
    lower=np.nextafter(z-t,-np.inf);upper=np.nextafter(z+t,np.inf)
    zi=[(lower[...,j,:3],upper[...,j,:3]) for j in range(3)]
    cs=[scale((x[0][...,0],x[1][...,0]),k) for x,k in zip(zi,costs)]
    dc,db=sub(cs[2],cs[0]),sub(cs[1],cs[0])
    dc=(dc[0][...,None],dc[1][...,None]);db=(db[0][...,None],db[1][...,None])
    aa=sub(mul(dc,sub(zi[1],zi[0])),mul(db,sub(zi[2],zi[0])))
    bb=sub(mul(dc,sub(scale(zi[1],costs[1]),scale(zi[0],costs[0]))),
           mul(db,sub(scale(zi[2],costs[2]),scale(zi[0],costs[0]))))
    coord=lambda x,n:(x[0][...,n],x[1][...,n])
    a0,a1,b1=coord(aa,0),coord(aa,1),coord(bb,1)
    dd=sub(mul(a0,sub(coord(bb,2),b1)),mul(a1,b1))
    defined=(dd[0]>0) & (b1[1]<0)
    vals=np.stack([b1[0]**2,b1[1]**2]);sqlo=vals.min(0)
    sqlo=np.where((b1[0]<=0)&(b1[1]>=0),0.,sqlo)
    squared=np.nextafter(sqlo,-np.inf),np.nextafter(vals.max(0),np.inf)
    ai=div(squared,dd);bi=div(mul(a0,b1),dd)
    alo=np.where(defined,np.maximum(0.,ai[0]),0.);ahi=np.where(defined,ai[1],np.inf)
    blo=np.where(defined,np.maximum(0.,bi[0]),0.);bhi=np.where(defined,np.minimum(1.,bi[1]),1.)
    return alo,ahi,blo,bhi,defined

def summary(z, costs, beta, sigma, m, rng, noise_correlation=.5):
    # Exact law of the average of m iid Gaussian episode noise vectors.
    # Within each episode, four lags have equicorrelation .5. Episodes independent.
    common = rng.normal(size=(REPS,3,1))
    independent = rng.normal(size=(REPS,3,4))
    noise = sigma/math.sqrt(m)*(math.sqrt(noise_correlation)*common+math.sqrt(1-noise_correlation)*independent)
    obs = z[None,:,:]+noise
    t = tail_threshold(sigma,m)
    eps = 4*t  # Four coefficients; simultaneous twelve-coordinate 95% event.
    ah,bh,rad,usable,valid,reject = estimate(obs,costs,eps)
    g=jacobian(obs,costs).reshape(REPS,2,3,4)
    variance=sigma**2/m*((1-noise_correlation)*(g*g).sum(axis=(2,3))+noise_correlation*(g.sum(-1)**2).sum(-1))
    se=np.sqrt(variance)
    critical=NormalDist().inv_cdf(1-DELTA/4)
    halfwidth=critical*se
    wald_cover=(np.abs(ah-ALPHA)<=halfwidth[:,0]) & (np.abs(bh-beta)<=halfwidth[:,1])
    alo,ahi,blo,bhi,box_defined=interval_confidence(obs,costs,t)
    box_cover=(alo<=ALPHA)&(ahi>=ALPHA)&(blo<=beta)&(bhi>=beta)
    cover = (~usable) | (np.maximum(np.abs(ah-ALPHA),np.abs(bh-beta)) <= rad)
    event=(np.max(np.abs(noise),axis=(1,2)) <= t)
    assert np.all(cover[event])
    assert not np.any(reject[event])
    assert np.all(box_cover[event])
    widths = np.where(usable,np.maximum(0,np.minimum(1,bh+rad)-np.maximum(0,bh-rad)),1)
    quant = lambda x,p: float(np.quantile(x,p))
    return {
        'm_per_environment':m,'sigma':sigma,'beta':beta,'replications':REPS,
        'alpha_rmse_all':float(np.sqrt(np.mean((ah-ALPHA)**2))),
        'beta_rmse_all':float(np.sqrt(np.mean((bh-beta)**2))),
        'alpha_median_abs_error':quant(np.abs(ah-ALPHA),.5),
        'beta_median_abs_error':quant(np.abs(bh-beta),.5),
        'beta_p95_abs_error':quant(np.abs(bh-beta),.95),
        'invalid_point_rate':float((~valid).mean()),
        'bounded_certificate_rate':float(usable.mean()),
        'beta_interval_narrower_than_domain_rate':float((widths < 1-1e-12).mean()),
        'beta_interval_median_width':quant(widths,.5),
        'confidence_set_coverage':float(cover.mean()),
        'coefficient_event_rate':float(event.mean()),
        'false_rejection_rate':float(reject.mean()),'epsilon_four_prefix':eps,
        'wald_simultaneous_coverage':float(wald_cover.mean()),
        'wald_beta_median_untruncated_width':float(np.median(2*halfwidth[:,1])),
        'response_box_coverage':float(box_cover.mean()),
        'response_box_bounded_rate':float(box_defined.mean()),
        'response_box_beta_median_width':float(np.median(np.maximum(0,bhi-blo))),
    }

def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    costs_set = {'wide':(.5,2,8),'ordinary':(1,2,4),'close':(1,1.1,1.2)}
    filters = {'direct':[1,0,0],'front':[.5,.35,.15],'weak':[.01,.09,.9]}
    rows=[]
    for cn,costs in costs_set.items():
        for fn,filt in filters.items():
            for beta in [0,2/3,.98]:
                z=response(ALPHA,beta,costs,POOL,filt)
                ah,bh,*_=estimate(z,costs,0.)
                assert abs(float(ah)-ALPHA)<1e-7 and abs(float(bh)-beta)<1e-7
                a,b,*_=terms(z,costs)
                residual=ALPHA*a[2]-beta*(b[3]-b[2])+b[2]-b[1]
                assert abs(float(residual))<1e-12
                for sigma in [1e-4,1e-2]:
                    for m in [100,1000,10000]:
                        row=summary(z,costs,beta,sigma,m,rng)
                        row.update(cost_design=cn,filter_design=fn,costs=list(costs),filter=filt)
                        rows.append(row)
    # Boundary audits of the observation contract, including zero response bias
    # from independently randomized pre-existing states at t=-1.
    scale_checks=[]
    for lam in [.1,10]:
        z=response(ALPHA,2/3,[lam*k for k in (1,2,4)],POOL,filters['front'])
        ah,bh,*_=estimate(z,(1,2,4),0.)
        assert abs(float(ah)-ALPHA/lam)<1e-10 and abs(float(bh)-2/3)<1e-10
        scale_checks.append({'unknown_common_cost_scale':lam,'alpha_in_relative_cost_units':float(ah),'beta_recovered':float(bh)})
    base=response(ALPHA,2/3,(1,2,4),POOL,filters['front'])
    corners=((np.arange(4096)[:,None] >> np.arange(12)) & 1)*2-1
    corners=corners.reshape(4096,3,4)
    corner_count=0
    for beta,cs,ff in [(0,(1,2,4),filters['direct']),
                       (2/3,(1,2,4),filters['front']),
                       (.98,(1,1.1,1.2),filters['front']),
                       (2/3,(1,2,4),filters['weak'])]:
        zz=response(ALPHA,beta,cs,POOL,ff)
        for eps in [1e-8,1e-6,1e-4]:
            perturbed=zz[None,:,:]+eps/4*corners
            ah,bh,rad,usable,valid,reject=estimate(perturbed,cs,eps)
            assert np.all(np.maximum(np.abs(ah[usable]-ALPHA),np.abs(bh[usable]-beta))<=rad[usable]+1e-9)
            assert not np.any(reject)
            alo,ahi,blo,bhi,defined=interval_confidence(perturbed,cs,eps/4*(1+1e-12))
            assert np.all((alo<=ALPHA)&(ahi>=ALPHA)&(blo<=beta)&(bhi>=beta))
            corner_count+=len(corners)
    analytic=jacobian(base,(1,2,4))
    for j in range(12):
        perturb=np.zeros((3,4),dtype=complex);perturb.flat[j]=1e-25j
        aa,bb,*_=terms(base.astype(complex)+perturb,(1,2,4))
        dd=aa[0]*(bb[2]-bb[1])-aa[1]*bb[1]
        numerical=np.imag(np.array([bb[1]**2/dd,aa[0]*bb[1]/dd]))/1e-25
        assert np.allclose(analytic[:,j],numerical,rtol=1e-10,atol=1e-10)
    delayed=np.pad(base,((0,0),(3,0)))
    assert np.all(delayed[:,:3]==0)
    assert np.allclose(delayed[:,3:],base,rtol=0,atol=0)
    # Paired impulse responses from identical, nonzero hidden initial states.
    pair_rows=[]
    for k in (1,2,4):
        r=1+2/3+ALPHA/k;e=2/(r+math.sqrt(r*r-4*(2/3)))
        outputs=[]
        for sign in [-1,1]:
            aggregate=np.zeros(4)
            for j,(s,weight) in enumerate(POOL):
                d,p=.3*(j+1),-.2*(j+1)
                gain=(1-e)*(1-(2/3)*e)/(1-(2/3)*e*s)
                for n in range(4):
                    d=s*d+(1-s)*(sign if n==0 else 0)
                    p=e*p+gain*d
                    aggregate[n]+=weight*p
            outputs.append(np.convolve(aggregate,filters['front'])[:4])
        pair_rows.append((outputs[1]-outputs[0])/2)
    assert np.allclose(np.array(pair_rows),base,rtol=1e-12,atol=1e-14)
    # Explicit raw-episode pilot, not only an aggregate sampling experiment.
    m=10000; amplitude=1.; sigma=1e-4
    signs=rng.choice([-1.,1.],size=(m,1,1))
    raw_noise=sigma*(math.sqrt(.5)*rng.normal(size=(m,3,1))+math.sqrt(.5)*rng.normal(size=(m,3,4)))
    pilot=np.mean(signs*(amplitude*signs*base[None,:,:]+raw_noise),axis=0)/amplitude
    t=tail_threshold(sigma,m)/amplitude
    ah,bh,rad,usable,valid,reject=estimate(pilot,(1,2,4),4*t)
    pilot_out={'m':m,'amplitude':amplitude,'sigma':sigma,'alpha_estimate':float(ah),'beta_estimate':float(bh),
               'bounded_certificate':bool(usable),'parameter_radius':float(rad) if usable else None,
               'max_coefficient_error':float(np.max(np.abs(pilot-base))),'coefficient_bound':t}
    # Misspecification power is reported separately; the coverage theorem does
    # not apply to these worlds. No invalid estimates are removed.
    mismatch=[]
    hetero=(response(1,0,(1,2,4),[(0,1)],[1,0,0])+response(4,0,(1,2,4),[(0,1)],[1,0,0]))
    changed_filter=base.copy();changed_filter[2]=response(ALPHA,2/3,(4,),POOL,[.7,.2,.1])[0]
    changed_pool=base.copy();changed_pool[2]=response(ALPHA,2/3,(4,),[(s,w*(1.3 if s>.5 else .8)) for s,w in POOL],filters['front'])[0]
    for label,z,true_beta in [('heterogeneous_alpha',hetero,0),('cost_specific_filter',changed_filter,2/3),('cost_specific_pool',changed_pool,2/3)]:
        for sigma in [1e-6,1e-3]:
            noise=sigma/100*(math.sqrt(.5)*rng.normal(size=(REPS,3,1))+math.sqrt(.5)*rng.normal(size=(REPS,3,4)))
            t=tail_threshold(sigma,10000)
            ah,bh,rad,usable,valid,reject=estimate(z[None,:,:]+noise,(1,2,4),4*t)
            mismatch.append({'failure':label,'sigma':sigma,'m_per_environment':10000,
                             'true_common_beta':true_beta,'median_apparent_beta':float(np.median(bh)),
                             'rejection_rate':float(reject.mean()),'bounded_certificate_rate':float(usable.mean()),
                             'invalid_point_rate':float((~valid).mean())})
    result={'seed':SEED,'replications_per_configuration':REPS,'configurations':len(rows),
            'total_correct_model_replications':len(rows)*REPS,'delta':DELTA,'alpha':ALPHA,'pool':POOL,
            'noise':'iid episodes; four Gaussian lags equicorrelated .5, independent environments; exact distribution of sample means',
            'amplitude':1,'scale_checks':scale_checks,'common_delay_check':3,'raw_episode_pilot':pilot_out,
            'jacobian_check':'Twelve complex-step derivative comparisons passed.',
            'bounded_noise_corner_checks':corner_count,
            'nonzero_initial_state_check':'Paired positive/negative episodes from identical nonzero target and policy states recover the zero-state kernel.',
            'results':rows,'misspecification':mismatch,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitations':['Synthetic calibrated fixed environments, not market or human preference data.',
                          'Noise proxy is supplied, not estimated. Four observed coefficients support residual checking.',
                          'Confidence sets use the full admissible parameter space when the sufficient inverse bound fails.',
                          'Monte Carlo coverage is not a proof of the concentration theorem or optimality.',
                          'Wald rectangles use known Gaussian noise covariance and are asymptotic, not finite-sample certificates.',
                          'Response-box algebra rounds outwards; an exact rational Taylor lower bound certifies the enlarged coefficient tail budget.',
                          'No invalid or extreme point estimates removed; beta=0 is a boundary case.']}
    (OUT/'statistical_results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    flat=[{k:v for k,v in r.items() if k not in ['costs','filter']} for r in rows]
    with (OUT/'statistical_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({'configurations':len(rows),'total_replications':len(rows)*REPS,'pilot':pilot_out,
                      'minimum_coverage':min(r['confidence_set_coverage'] for r in rows),
                      'maximum_false_rejection_rate':max(r['false_rejection_rate'] for r in rows),
                      'misspecification':mismatch},ensure_ascii=False))

if __name__=='__main__':main()
