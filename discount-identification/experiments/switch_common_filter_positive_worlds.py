from fractions import Fraction as F
from pathlib import Path
from math import comb
import json
root=Path(__file__).resolve().parent
def trim(p):
    p=list(p)
    while len(p)>1 and p[-1]==0:p.pop()
    return p
def add(p,q):
    a=[F(0)]*max(len(p),len(q))
    for j,v in enumerate(p):a[j]+=v
    for j,v in enumerate(q):a[j]+=v
    return trim(a)
def scale(p,c):return trim([c*x for x in p])
def mul(p,q):
    a=[F(0)]*(len(p)+len(q)-1)
    for j,v in enumerate(p):
        for l,w in enumerate(q):a[j+l]+=v*w
    return trim(a)
def val(p,x):
    a=F(0)
    for v in p[::-1]:a=a*x+v
    return a
def deriv(p):return [j*p[j] for j in range(1,len(p))]
def bern_interval(p,lo,hi):
    n=len(p)-1
    a=[sum(p[j]*comb(j,k)*lo**(j-k)*(hi-lo)**k for j in range(k,n+1)) for k in range(n+1)]
    return [sum(a[j]*F(comb(k,j),comb(n,j)) for j in range(k+1)) for k in range(n+1)]
def isolate(p,lo,hi):
    vl=val(p,lo);vh=val(p,hi);assert vl*vh<0
    for _ in range(48):
        mid=(lo+hi)/2;vm=val(p,mid)
        if vm==0:return (mid,mid)
        if vm*vl>0:lo=mid;vl=vm
        else:hi=mid;vh=vm
    return lo,hi
b0=F(2,3);alpha=F(5,2);ea=F(1,4);eb=F(3,8)
ss=[F(0),F(1,10),F(3,10),F(7,10)]
def gain(e,s):return (1-e)*(1-b0*e)*(1-s)/(1-b0*e*s)
def exc(e,s):return -gain(e,s)*e/(s-e)
w0=F(1,100);w3=F(1)
a11,a12=exc(ea,ss[1]),exc(ea,ss[2])
a21,a22=exc(eb,ss[1]),exc(eb,ss[2])
r1=-w0*exc(ea,F(0))-w3*exc(ea,ss[3])
r2=-w0*exc(eb,F(0))-w3*exc(eb,ss[3])
det=a11*a22-a12*a21
w1=(r1*a22-a12*r2)/det;w2=(a11*r2-r1*a21)/det
weights=[w0,w1,w2,w3];assert all(w>0 for w in weights)
assert all(sum(w*exc(e,s) for w,s in zip(weights,ss))==0 for e in [ea,eb])
Q=[F(1)]
for s in ss[1:]:Q=mul(Q,[F(1),-s])
P=[]
for e in [ea,eb]:
    p=[F(0)]
    for j,s in enumerate(ss[1:],start=1):
        q=[F(1)]
        for s2 in ss[1:]:
            if s2!=s:q=mul(q,[F(1),-s2])
        residue=weights[j]*gain(e,s)*s/(s-e)
        p=add(p,scale(q,residue))
    P.append(p)
def L(k,b):return [ -k*b,alpha+k*(1+b),-k]
for e,p in zip([ea,eb],P):
    for z in [F(0),F(1,5),F(1,2),F(1)]:
        direct=sum(w*gain(e,s)/((1-e*z)*(1-s*z)) for w,s in zip(weights,ss))
        assert direct==val(p,z)/val(Q,z)
C=val(add(mul(L(2,b0),P[1]),scale(mul(L(1,b0),P[0]),-1)),F(0))
assert add(mul(L(2,b0),P[1]),scale(mul(L(1,b0),P[0]),-1))==scale(Q,C)
Fa1=val(P[0],F(1))/val(Q,F(1))
def make_world(b):
    qq=add(mul(L(2,b),P[1]),scale(mul(L(1,b),P[0]),-1))
    assert len(qq)==4 and qq[0]!=0
    qq=scale(qq,1/qq[0])
    xx=add(mul(L(1,b),P[0]),scale(qq,b*P[0][0]))
    assert xx[0]==0
    nx=scale(xx[1:],1/alpha)
    atom0=nx[-1]/qq[-1];assert atom0>0
    rp=qq[::-1];tt=nx[::-1]
    intervals=[]
    for center,sgn in zip(ss[1:],[1,-1,1]):
        lo,hi=isolate(rp,center-F(1,100),center+F(1,100))
        dp=bern_interval(deriv(rp),lo,hi);bp=bern_interval(tt,lo,hi)
        assert all(x*sgn>0 for x in dp) and all(x*sgn>0 for x in bp)
        intervals.append(dict(lo=str(lo),hi=str(hi),derivative_bern=[str(x) for x in dp],target_bern=[str(x) for x in bp]))
    # Same ratio exactly. Each world is rescaled to the SAME Fa(1), enabling unit-gain filters.
    rawFa1=val(P[0],F(1))/val(qq,F(1));lam=Fa1/rawFa1;assert lam>0
    def X(z):return lam*val(nx,z)/val(qq,z)
    # Future fee 4: root brackets and positivity imply nu(E) strictly decreasing.
    lo=F(0);hi=F(1)
    for _ in range(64):
        mid=(lo+hi)/2
        polynomial=b*mid*mid-(1+b+alpha/4)*mid+1
        if polynomial>0:lo=mid
        else:hi=mid
    def nu(e):return X(F(1))-b*e*X(b*e)
    lower=nu(hi)-Fa1;upper=nu(lo)-Fa1;assert lower<=upper
    return dict(beta=str(b),Q=[str(v) for v in qq],target_numerator=[str(v) for v in nx],
        atom0_before_scale=str(atom0),scale_exact=str(lam),positive_target_intervals=intervals,
        future_E_interval=[str(lo),str(hi)],Gamma_interval=[str(lower),str(upper)],
        Gamma_mid_display=float((lower+upper)/2))
worlds=[make_world(b0),make_world(b0+F(1,10000))]
a,b=worlds
al,ah=map(F,a["Gamma_interval"]);bl,bh=map(F,b["Gamma_interval"])
assert ah<bl or bh<al
out=dict(label="E149",status="exact_positive_target_world_certificates_and_separated_future_functional",
 alpha_exact=str(alpha),fees=["1","2"],future_fee="4",baseline_targets=[str(s) for s in ss],
 baseline_weights=[str(w) for w in weights],P_a=[str(x) for x in P[0]],P_b=[str(x) for x in P[1]],
 shared_Fa_one_exact=str(Fa1),worlds=worlds,
 observation_identity="K_A=Fa_B/Fa_A(1); K_B=Fa_A/Fa_A(1). Each common filter nonnegative stable unit gain. Both channels exactly identical by shared P_b/P_a.",
 limitations="Algebraic roots defined by certified disjoint rational intervals; no float positivity proof. Euler-policy equivalence and complete written proof must still be audited. Not empirical prices, institutions or capital.")
(root/"switch_common_filter_positive_worlds_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({"label":"E149","baseline_weights":[str(w) for w in weights],"betas":[x["beta"] for x in worlds],"Gamma_display":[x["Gamma_mid_display"] for x in worlds],"separated_exact_intervals":True}))
