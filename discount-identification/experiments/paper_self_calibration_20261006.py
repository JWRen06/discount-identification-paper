"""Explore and verify locally identifiable joint cost/discount recovery.

Costs are not supplied as exact ratios. The auditor estimates alpha/kappa_a,
beta, kappa_b/kappa_a and kappa_c/kappa_a from observed prefix restrictions.
Finite multistart solutions do not certify global uniqueness.
"""
from pathlib import Path
import json,itertools
from fractions import Fraction as F
import numpy as np
from independent_tracking_controller_20261006 import paired_impulse
from paper_calibration_audit_20261006 import add,sub,mul,div

OUT=Path(__file__).resolve().parent/'paper_breakthrough_20261006'
POOL=[(0.,.2),(.1,.3),(.7,1.),(.95,.5)]

def residual(x,z):
    a,b,r2,r3=x;k=np.array([1,r2,r3]);c=k*z[:,0]
    aa=(c[2]-c[0])*(z[1]-z[0])-(c[1]-c[0])*(z[2]-z[0])
    bb=(c[2]-c[0])*(k[1]*z[1]-k[0]*z[0])-(c[1]-c[0])*(k[2]*z[2]-k[0]*z[0])
    u=bb.copy();u[1:]-=bb[:-1]
    return a*aa[:-1]-b*u[1:]+u[:-1]

def jacobian(x,z):
    return np.stack([residual(np.asarray(x,dtype=complex)+1e-25j*np.eye(4)[j],z).imag/1e-25 for j in range(4)],1)

def pencil(z):
    """Six-column structured annihilator, coefficients at z^1,...,z^(L-1)."""
    a0,b0,c0=z[:,0]
    aa=a0*(z[2]-z[1]);ab=b0*(z[0]-z[2])
    p=b0*z[0]-a0*z[1];q=a0*z[2]-c0*z[0]
    p[0]=q[0]=0.
    def shift(v):return np.r_[0.,v[:-1]]
    return np.stack([shift(aa),shift(ab),shift(p-shift(p)),shift(q-shift(q)),
                     -(p-shift(p)),-(q-shift(q))],1)[1:]

def closed_recover(z):
    C=pencil(z);_,sv,V=np.linalg.svd(C,full_matrices=True);w=V[-1]
    a0,b0,c0=z[:,0];t=w[1]/w[0]
    h=w[3]/w[2]/(1-t*b0/a0)
    r2=t/(1-h*c0/b0+t*h*c0/a0);r3=h*r2
    alpha=w[0]/w[2]*r2;beta=w[4]/w[2]
    return np.array([alpha,beta,r2,r3]),sv,w

def interval_pencil(z,t):
    lo=np.nextafter(z-t,-np.inf);hi=np.nextafter(z+t,np.inf)
    zz=[(lo[j],hi[j]) for j in range(3)]
    z0=[(lo[j,0],hi[j,0]) for j in range(3)]
    aa=mul(z0[0],sub(zz[2],zz[1]));ab=mul(z0[1],sub(zz[0],zz[2]))
    p=sub(mul(z0[1],zz[0]),mul(z0[0],zz[1]))
    q=sub(mul(z0[0],zz[2]),mul(z0[2],zz[0]))
    for v in [p,q]:v[0][0]=v[1][0]=0.
    def shift(v):return (np.r_[0.,v[0][:-1]],np.r_[0.,v[1][:-1]])
    pp=sub(p,shift(p));qq=sub(q,shift(q))
    cols=[shift(aa),shift(ab),shift(pp),shift(qq),(-pp[1],-pp[0]),(-qq[1],-qq[0])]
    return np.stack([c[0] for c in cols],1)[1:],np.stack([c[1] for c in cols],1)[1:]

def self_calibration_certificate(z,t):
    """Cost-free beta interval via a certified perturbed 5x5 linear system.

    Fix structured null-vector component p=1. Select five observed rows, then
    use an a-posteriori inverse residual bound rather than assuming an exact
    floating inverse. On a simultaneous response box, inclusion is deterministic.
    """
    C=pencil(z);cl,ch=interval_pencil(z,t);cols=[0,1,3,4,5]
    D=C[:,cols];_,_,V=np.linalg.svd(D,full_matrices=False)
    # Greedy row pivoting: choose linearly independent, well-scaled rows.
    available=list(range(len(D)));rows=[];basis=[]
    for _ in range(5):
        best=None;bestnorm=-1.
        for j in available:
            v=D[j].copy()
            for q in basis:v-=np.dot(v,q)*q
            score=float(v@v)
            if score>bestnorm:best,bestnorm,vector=j,score,v
        if bestnorm<=1e-30:return dict(beta_interval=[0.,1.],informative=False,reason='rank gap absent')
        rows.append(best);available.remove(best);basis.append(vector/np.sqrt(bestnorm))
    A=D[rows];b=-C[rows,2];AL=cl[np.ix_(rows,cols)];AH=ch[np.ix_(rows,cols)]
    try:R=np.linalg.inv(A);x=R@b
    except np.linalg.LinAlgError:return dict(beta_interval=[0.,1.],informative=False,reason='singular pivot')
    # Interval enclosure of RA; includes each product and sum roundoff.
    lower=np.zeros((5,5));upper=np.zeros((5,5))
    for j in range(5):
        product=mul((R[:,j,None],R[:,j,None]),(AL[j,None,:],AH[j,None,:]))
        lower,upper=add((lower,upper),product)
    diff=sub((np.eye(5),np.eye(5)),(lower,upper))
    def upper_row_sum(v):
        total=np.zeros(v.shape[0])
        for j in range(v.shape[1]):total=np.nextafter(total+v[:,j],np.inf)
        return float(total.max())
    eps=upper_row_sum(np.maximum(abs(diff[0]),abs(diff[1])))
    if not eps<1:return dict(beta_interval=[0.,1.],informative=False,reason='inverse perturbation too large',inverse_defect=float(eps))
    bl,bh=-ch[rows,2],-cl[rows,2]
    acc=(bl.copy(),bh.copy())
    for j in range(5):acc=sub(acc,mul((AL[:,j],AH[:,j]),(x[j],x[j])))
    rb=np.nextafter(np.maximum(abs(acc[0]),abs(acc[1])).max(),np.inf)
    Rnorm=upper_row_sum(abs(R))
    denominator=np.nextafter(1-eps,-np.inf)
    radius=np.nextafter(np.nextafter(Rnorm*rb,np.inf)/denominator,np.inf)
    interval=[max(0.,float(np.nextafter(x[3]-radius,-np.inf))),min(1.,float(np.nextafter(x[3]+radius,np.inf)))]
    if interval[0]>interval[1]:return dict(beta_interval=[0.,1.],informative=False,reason='model or bound incompatible')
    return dict(beta_interval=interval,informative=interval!=[0.,1.],inverse_defect=float(eps),
        vector_radius=float(radius),selected_rows=rows,normalized_vector=x.tolist(),
        beta=float(x[3]),bound_half_width=t)

def exact_witness():
    beta=F(2,3);alpha=F(5,2);Es=[F(1,4),F(2,5),F(3,5)]
    pool=[(F(0),F(1,5)),(F(1,10),F(3,10)),(F(7,10),F(1)),(F(19,20),F(1,2))]
    filt=[F(1,2),F(7,20),F(3,20)];L=8;z=[]
    for E in Es:
        v=[sum(w*(1-E)*(1-beta*E)*(1-s)/(1-beta*E*s)*sum(E**j*s**(n-j) for j in range(n+1)) for s,w in pool) for n in range(L)]
        z.append([sum(filt[j]*v[n-j] for j in range(min(n+1,len(filt)))) for n in range(L)])
    a,b,c=[v[0] for v in z]
    aa=[a*(z[2][n]-z[1][n]) for n in range(L)];ab=[b*(z[0][n]-z[2][n]) for n in range(L)]
    p=[b*z[0][n]-a*z[1][n] for n in range(L)];q=[a*z[2][n]-c*z[0][n] for n in range(L)]
    shift=lambda v:[F(0)]+v[:-1]
    pp=[x-y for x,y in zip(p,shift(p))];qq=[x-y for x,y in zip(q,shift(q))]
    C=[list(row) for row in zip(shift(aa),shift(ab),shift(pp),shift(qq),[-x for x in pp],[-x for x in qq])][1:]
    rows=[r[:] for r in C];pivots=[];i=0
    for j in range(6):
        pivot=next((k for k in range(i,len(rows)) if rows[k][j]),None)
        if pivot is None:continue
        rows[i],rows[pivot]=rows[pivot],rows[i];scale=rows[i][j];rows[i]=[x/scale for x in rows[i]]
        for k in range(len(rows)):
            if k!=i:
                scale=rows[k][j];rows[k]=[x-scale*y for x,y in zip(rows[k],rows[i])]
        pivots.append(j);i+=1
        if i==len(rows):break
    assert len(pivots)==5
    free=next(j for j in range(6) if j not in pivots);w=[F(0)]*6;w[free]=F(1)
    for i,j in enumerate(pivots):w[j]=-rows[i][free]
    w=[x/w[2] for x in w];t=w[1]/w[0];h=w[3]/(1-t*b/a)
    r2=t/(1-h*c/b+t*h*c/a);r3=h*r2;recovered=[w[0]*r2,w[4],r2,r3]
    costs=[alpha*E/((1-E)*(1-beta*E)) for E in Es]
    assert recovered==[alpha/costs[0],beta,costs[1]/costs[0],costs[2]/costs[0]]
    return dict(rank=5,arithmetic='exact Fraction Gaussian elimination; no tolerance',recovered=[str(x) for x in recovered],
                true_costs=[str(x) for x in costs],null_vector=[str(x) for x in w])

def fit(z,start,bounds=((.05,30.),(0.,.999999),(1.25,3.),(3.25,6.))):
    x=np.array(start,dtype=float);lo,hi=np.array(bounds).T
    x=np.clip(x,lo+1e-10,hi-1e-10)
    scale=max(float(np.max(abs(z)))**2,1e-8)
    for iteration in range(200):
        r=residual(x,z)/scale;J=jacobian(x,z)/scale
        step=np.linalg.lstsq(J,-r,rcond=1e-12)[0]
        old=float(r@r);accepted=False
        for h in range(22):
            xn=np.clip(x+step*2.**(-h),lo+1e-10,hi-1e-10)
            rn=residual(xn,z)/scale;new=float(rn@rn)
            if new<old:
                accepted=True;break
        if not accepted:break
        change=float(np.max(abs(xn-x)));x=xn
        if change<1e-11 or new<1e-26:break
    return x,float(np.linalg.norm(residual(x,z)/scale)),iteration+1

def main():
    OUT.mkdir(exist_ok=True);rng=np.random.default_rng(2026100604)
    ranks=[]
    for beta,pool,label,L in itertools.product([0.,.3,2/3,.98],[[(0.,1.)],[(.9,1.)],POOL],['front'],[4,5,8,24]):
        z,_=paired_impulse(2.5,beta,[1.,2.,4.],pool,[.5,.35,.15],L)
        sv=np.linalg.svd(jacobian([2.5,beta,2.,4.],z),compute_uv=False)
        ps=np.linalg.svd(pencil(z),compute_uv=False)
        ranks.append(dict(beta=beta,pool_size=len(pool),L=L,singular_values=sv.tolist(),rank=int(np.sum(sv>sv[0]*1e-10)),
                          pencil_singular_values=ps.tolist(),pencil_rank=int(np.sum(ps>ps[0]*1e-10))))
    true,_=paired_impulse(2.5,2/3,[1.,2.,4.],POOL,[.5,.35,.15],16)
    closed,sv,w=closed_recover(true)
    assert np.max(abs(closed-[2.5,2/3,2,4]))<1e-8
    starts=np.column_stack([rng.uniform(.5,8,32),rng.uniform(.05,.95,32),rng.uniform(1.25,3,32),rng.uniform(3.25,6,32)])
    multistart=[dict(start=s.tolist(),estimate=x.tolist(),scaled_residual=rr,iterations=it) for s in starts for x,rr,it in [fit(true,s)]]
    recovered=int(sum(np.max(abs(np.array(x['estimate'])-[2.5,2/3,2,4]))<1e-6 for x in multistart))
    assert recovered>0
    noisy=[]
    for L,m,sigma in itertools.product([5,8,16],[1000,10000],[1e-4]):
        base=true[:,:L];est=[]
        for _ in range(300):
            e=sigma/np.sqrt(m)*(np.sqrt(.5)*rng.normal()+np.sqrt(.5)*rng.normal(size=base.shape))
            candidates=[fit(base+e,s) for s in [[2.5,.5,2.02,4],[1.,.9,1.5,5.],[5.,.1,2.8,3.5]]]
            est.append(min(candidates,key=lambda row:row[1])[0])
        est=np.array(est)
        noisy.append(dict(L=L,m=m,sigma=sigma,repetitions=len(est),beta_median_absolute_error=float(np.median(abs(est[:,1]-2/3))),
            median_parameter_error=np.median(abs(est-[2.5,2/3,2,4]),axis=0).tolist(),
            beta_quantiles=np.quantile(est[:,1],[.025,.5,.975]).tolist()))
    certificates=[]
    for L,t in itertools.product([6,8,16],[1e-6,1e-8,1e-10,1e-12]):
        ci=self_calibration_certificate(true[:,:L],t)
        assert ci['beta_interval'][0]<=2/3<=ci['beta_interval'][1]
        certificates.append(dict(L=L,**ci))
    perturbed_checks=0
    for beta,filt in itertools.product([0.,2/3],[[.5,.35,.15],[.01,.09,.9]]):
        z,_=paired_impulse(2.5,beta,[1.,2.,4.],POOL,filt,16)
        for _ in range(64):
            t=1e-10;obs=z+rng.uniform(-t,t,size=z.shape);ci=self_calibration_certificate(obs,t)
            assert ci['beta_interval'][0]<=beta<=ci['beta_interval'][1];perturbed_checks+=1
    result=dict(seed=2026100604,rank_checks=ranks,noiseless_multistart=multistart,noisy_recovery=noisy,
        closed_recovery=closed.tolist(),pencil_singular_values=sv.tolist(),cost_free_certificates=certificates,
        exact_rational_witness=exact_witness(),perturbed_cost_free_coverage_checks=perturbed_checks,
        observation_source='independent Bellman paired replay',parameter_order=['alpha/reference_cost','beta','cost_ratio_b','cost_ratio_c'],
        limitations=['Rank implies local uniqueness only.','Multistart is not a proof of global uniqueness.',
                    'Point estimation has no finite-sample coverage guarantee.','Admissible separated cost ranges are supplied, not exact ratios.'])
    (OUT/'self_calibration_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(ranks=[(x['beta'],x['pool_size'],x['L'],x['rank']) for x in ranks],recovered_starts=recovered,
        other_starts=[x for x in multistart if np.max(abs(np.array(x['estimate'])-[2.5,2/3,2,4]))>=1e-6],noise=noisy)))

if __name__=='__main__':main()
