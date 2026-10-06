"""Certified componentwise recovery: all rows, no cost inputs.

Historical certificates and logs remain unchanged. All arithmetic bounds use
outward interval operations; numerical optimization only chooses preconditioners.
"""
from pathlib import Path
import json, hashlib,itertools
from fractions import Fraction as F
import numpy as np
from paper_self_calibration_20261006 import pencil,interval_pencil,self_calibration_certificate
from paper_calibration_audit_20261006 import add,sub,mul

ROOT=Path(__file__).resolve().parent/'paper_breakthrough_20261006'
OUT=Path(__file__).resolve().parent/'paper_stability_revision_20261006'

def rational_generic_witness():
    beta=F(2,3);Es=[F(1,4),F(2,5),F(3,5)];L=6
    pool=[(F(1,10),F(1)),(F(7,10),F(1))];z=[]
    for E in Es:z.append([sum(w*(1-E)*(1-beta*E)*(1-s)/(1-beta*E*s)*sum(E**j*s**(n-j) for j in range(n+1)) for s,w in pool) for n in range(L)])
    C=fraction_pencil(z);D=[[r[j] for j in [0,1,3,4,5]] for r in C];det=F(1)
    for j in range(5):
        k=next(k for k in range(j,5) if D[k][j])
        if k!=j:D[k],D[j]=D[j],D[k];det=-det
        v=D[j][j];det*=v
        for k in range(j+1,5):
            fac=D[k][j]/v
            for h in range(j,5):D[k][h]-=fac*D[j][h]
    assert det!=0
    return dict(beta=str(beta),alpha='5/2',execution_roots=list(map(str,Es)),costs=['1','25/11','25/4'],pool=[['1/10','1'],['7/10','1']],L=L,minor_columns=[0,1,3,4,5],determinant=str(det))

def fraction_pencil(z):
    L=len(z[0]);a,b,c=[v[0] for v in z]
    aa=[a*(z[2][n]-z[1][n]) for n in range(L)];ab=[b*(z[0][n]-z[2][n]) for n in range(L)]
    p=[b*z[0][n]-a*z[1][n] for n in range(L)];q=[a*z[2][n]-c*z[0][n] for n in range(L)]
    shift=lambda v:[F(0)]+v[:-1]
    pp=[x-y for x,y in zip(p,shift(p))];qq=[x-y for x,y in zip(q,shift(q))]
    return [list(row) for row in zip(shift(aa),shift(ab),shift(pp),shift(qq),[-x for x in pp],[-x for x in qq])][1:]

def mat_interval(R,L,H):
    low=np.zeros((R.shape[0],L.shape[1]));high=low.copy()
    for j in range(R.shape[1]):
        low,high=add((low,high),mul((R[:,j,None],R[:,j,None]),(L[j,None,:],H[j,None,:])))
    return low,high

def upper_sum(x):
    s=np.zeros(x.shape[0])
    for j in range(x.shape[1]):s=np.nextafter(s+x[:,j],np.inf)
    return s

def structured_product(R,w,z,t):
    """Enclose RH(z)w with shared polynomial coefficients collected first.

    Each output is sum_j h_j times a linear form in all response entries.
    Crucially, coefficients for the same noisy entry are combined before boxing.
    Every coefficient operation is itself interval-enclosed.
    """
    m,L=z.shape;out=R.shape[0]
    cl=np.zeros((out,3,3,L));ch=cl.copy()
    def term(group,signal,lag,weight):
        nonlocal cl,ch
        for row in range(R.shape[1]):
            n=row+1-lag
            if n<0:continue
            a,b=mul((R[:,row],R[:,row]),(weight,weight))
            cl[:,group,signal,n],ch[:,group,signal,n]=add((cl[:,group,signal,n],ch[:,group,signal,n]),(a,b))
    # ha
    term(0,2,1,w[0]);term(0,1,1,-w[0])
    term(0,1,1,-w[2]);term(0,1,2,w[2])
    term(0,2,1,w[3]);term(0,2,2,-w[3])
    term(0,1,0,w[4]);term(0,1,1,-w[4])
    term(0,2,0,-w[5]);term(0,2,1,w[5])
    # hb
    term(1,0,1,w[1]);term(1,2,1,-w[1])
    term(1,0,1,w[2]);term(1,0,2,-w[2])
    term(1,0,0,-w[4]);term(1,0,1,w[4])
    # hc
    term(2,0,1,-w[3]);term(2,0,2,w[3])
    term(2,0,0,w[5]);term(2,0,1,-w[5])
    zl,zh=np.nextafter(z-t,-np.inf),np.nextafter(z+t,np.inf)
    low=np.zeros(out);high=low.copy()
    for group in range(3):
        a=np.zeros(out);b=a.copy()
        for signal in range(3):
            for n in range(L):a,b=add((a,b),mul((cl[:,group,signal,n],ch[:,group,signal,n]),(zl[signal,n],zh[signal,n])))
        low,high=add((low,high),mul((zl[group,0],zh[group,0]),(a,b)))
    return low,high

def certificate(z,t):
    if not np.all(np.isfinite(z)) or not np.isfinite(t) or t<0 or z.shape[1]<6:
        return dict(beta_interval=[0.,1.],informative=False,reason='invalid observation or error budget')
    C=pencil(z);L,H=interval_pencil(z,t);cols=[0,1,3,4,5]
    D=C[:,cols];DL=L[:,cols];DH=H[:,cols];b=-C[:,2]
    # Use the entire observed prefix. Column scaling is only a numerical choice.
    scale=np.maximum(np.linalg.norm(D,axis=0),1e-300)
    try:R=np.linalg.pinv(D/scale,rcond=1e-15)/scale[:,None]
    except np.linalg.LinAlgError:return dict(beta_interval=[0.,1.],informative=False,reason='preconditioner failed')
    x=R@b
    rdl=np.zeros((5,5));rdh=rdl.copy()
    for j,col in enumerate(cols):
        w=np.eye(6)[col];rdl[:,j],rdh[:,j]=structured_product(R,w,z,t)
    el,eh=sub((np.eye(5),np.eye(5)),(rdl,rdh))
    M=np.nextafter(np.maximum(abs(el),abs(eh)),np.inf)
    # Enclose the preconditioned residual directly, not ||R|| times one worst residual.
    w=np.array([x[0],x[1],1.,x[2],x[3],x[4]])
    rl,rh=structured_product(R,w,z,t)
    g=np.nextafter(np.maximum(abs(rl),abs(rh)),np.inf)
    # A positive supersolution v provides weighted contraction and a component bound.
    try:v=np.linalg.solve(np.eye(5)-M,g)
    except np.linalg.LinAlgError:v=np.full(5,-1.)
    if np.all(np.isfinite(v)) and np.all(v>0):
        v=np.nextafter(v*1.000001+1e-30,np.inf)
        mv=upper_sum(np.nextafter(M*v[None,:],np.inf))
        if np.all(np.nextafter(g+mv,np.inf)<=v):
            ratio=float(np.max(np.nextafter(mv/v,np.inf)))
            if ratio<1:
                ci=[max(0.,float(np.nextafter(x[3]-v[3],-np.inf))),min(1.,float(np.nextafter(x[3]+v[3],np.inf)))]
                if ci[0]<=ci[1]:return dict(beta_interval=ci,informative=ci!=[0.,1.],center=x.tolist(),component_radii=v.tolist(),weighted_contraction=ratio,rows=len(D),error_half_width=t)
    return dict(beta_interval=[0.,1.],informative=False,reason='no verified positive supersolution',rows=len(D),error_half_width=t)

def main():
    OUT.mkdir(exist_ok=True)
    path=ROOT/'external_paired_observations.json';logs=json.loads(path.read_text());results=[]
    for record in logs:
        raw=np.asarray(record['paired_reported_aggregate_positions'])
        z=(raw[:,1]-raw[:,0])/(2*record['input_amplitude'])
        for t in [1.1e-12,1.1e-11,1.1e-10,1.1e-9,1.1e-8,1.1e-7,1.1e-6]:
            old=self_calibration_certificate(z,t);new=certificate(z,t)
            results.append(dict(label=record['replay_label'],old=old,new=new))
    # Verify polynomial interval inclusion against independently expanded exact fractions.
    rng=np.random.default_rng(2026100609);exact_checks=0
    for _ in range(24):
        z=rng.normal(size=(3,8));R=rng.normal(size=(5,7));w=rng.normal(size=6);t=1e-5
        perturbed=z+rng.uniform(-t,t,z.shape);lo,hi=structured_product(R,w,z,t)
        C=fraction_pencil([[F(float(v)) for v in row] for row in perturbed])
        for i in range(5):
            truth=sum(F(float(R[i,j]))*sum(C[j][k]*F(float(w[k])) for k in range(6)) for j in range(7))
            assert F(float(lo[i]))<=truth<=F(float(hi[i]));exact_checks+=1
    from paper_external_rule_replay_20261006 import paired_prefix,official_prices
    pre=official_prices()['WTI_EIA'][0];baseline=np.diff([p for d,p in pre])[-64:];baseline=(baseline-baseline.mean())/baseline.std()
    # Designs were explored on existing cases; validation is not advertised as unseen research.
    designs=[];newlogs=[];coverage=[]
    for costs,beta in itertools.product([[1.,2.,4.],[.01,.1,1.]], [.8,.9999]):
        z,raw,t=paired_prefix(.9999/15,beta,costs,L=64,report_quantum=1e-8,baseline=baseline)
        ci=certificate(z,t)
        assert ci['beta_interval'][0]<=beta<=ci['beta_interval'][1]
        designs.append(dict(private_costs=costs,private_beta=beta,certificate=ci))
        newlogs.append(dict(label=f'case_{len(newlogs)+1}',cost_labels=['A','B','C'],paired_reports=raw.tolist(),input_amplitude=1.,response_error_bound=t,baseline_period='last 64 development observations, ending no later than 2022-05-23'))
        for noise in [1e-8,1e-6]:
            checks=0;determinate=0
            for _ in range(16):
                zz=z+rng.uniform(-noise,noise,z.shape);cc=certificate(zz,t+noise)
                assert cc['beta_interval'][0]<=beta<=cc['beta_interval'][1];checks+=1
                determinate+=int(cc['beta_interval'][0]>=.99 or cc['beta_interval'][1]<.99)
            coverage.append(dict(costs=costs,beta=beta,added_bounded_noise=noise,coverage_checks=checks,determinate=determinate))
    noise_proxy=1e-4;delta=.05;N=192;sample=[]
    for A in [1.,100.,1000.]:
        m=int(np.ceil(2*(noise_proxy/(A*1e-8))**2*np.log(2*N/delta)))
        sample.append(dict(subgaussian_response_noise_proxy=noise_proxy,amplitude=A,target_t=1e-8,delta=delta,coefficients=N,required_repetitions=m))
    (OUT/'noisy_paired_observations.json').write_text(json.dumps(newlogs,indent=2)+'\n',encoding='utf-8')
    result=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),method='all-row shared-polynomial componentwise verified supersolution',comparisons=results,
                generic_rank_witness=rational_generic_witness(),exact_interval_product_checks=exact_checks,design_comparisons=designs,perturbation_checks=coverage,sample_amplitude_tradeoff=sample,
                limitations=['Designs explored on previously observed configurations, not independent external validation.','Costs supplied to producer only; audit uses response labels.','Quantization noise is not market measurement noise.','Amplitude improvement requires linear unsaturated contract over the enlarged range.'])
    (OUT/'stability_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([(r['label'],r['new']['error_half_width'],r['new']['beta_interval']) for r in results]))

if __name__=='__main__':main()
