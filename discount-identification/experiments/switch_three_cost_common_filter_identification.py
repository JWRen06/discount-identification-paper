from fractions import Fraction as F
from pathlib import Path
import json
root=Path(__file__).resolve().parent
old=json.loads((root/"switch_common_filter_positive_worlds_results.json").read_text())
weights=list(map(F,old["baseline_weights"]));ss=list(map(F,old["baseline_targets"]))
def conv(a,b,N):return [sum(a[j]*b[n-j] for j in range(n+1) if j<len(a) and n-j<len(b)) for n in range(N+1)]
def add(a,b,x=F(1),y=F(1)):return [x*v+y*w for v,w in zip(a,b)]
def coeff(e,b,targets,masses,N):
    return [sum(w*(1-e)*(1-b*e)*(1-s)/(1-b*e*s)*sum(e**j*s**(n-j) for j in range(n+1)) for s,w in zip(targets,masses)) for n in range(N+1)]
def check(b,alpha,es,ks,targets,masses,K,N=12):
    Fs=[coeff(e,b,targets,masses,N) for e in es]
    Z=[conv(K,f,N) for f in Fs];c=[k*z[0] for k,z in zip(ks,Z)]
    assert all(c[j+1]>c[j] for j in range(2)) if K[0]>0 else True
    A=add(add(Z[1],Z[0],F(1),F(-1)),add(Z[2],Z[0],F(1),F(-1)),c[2]-c[0],-(c[1]-c[0]))
    B=add(add(Z[1],Z[0],ks[1],-ks[0]),add(Z[2],Z[0],ks[2],-ks[0]),c[2]-c[0],-(c[1]-c[0]))
    poly=conv([F(1),F(-1)],[ -b,F(1)],2)
    residual=add([F(0)]+A[:-1],conv(poly,B,N),alpha,F(1))
    assert all(v==0 for v in residual)
    m=next(j for j,v in enumerate(B) if v)
    num=[F(0)]+[-v for v in A[:-1]]
    den=conv([F(1),F(-1)],B,N)
    assert all(v==0 for v in num[:m])
    C0=num[m]/den[m];C1=(num[m+1]-C0*den[m+1])/den[m]
    assert 1/C1==alpha and -C0/C1==b
    return dict(beta=str(b),alpha=str(alpha),fees=[str(k) for k in ks],filter=[str(x) for x in K],first_nonzero_B=m,required_last_index=m+1,
        C0=str(C0),C1=str(C1),recovered_beta=str(-C0/C1),recovered_alpha=str(1/C1),all_prefix_identity_checks=True)
rows=[]
for b,alpha,es in [(F(2,3),F(5,2),[F(1,4),F(3,8),F(1,2)]),(F(0),F(5,2),[F(1,4),F(3,8),F(1,2)])]:
    ks=[alpha*e/((1-e)*(1-b*e)) for e in es]
    for ts,ws in [(ss,weights),([F(0)],[F(1)]),([F(1,2),F(9,10)],[F(2,3),F(1,3)])]:
        for K in [[F(1)],[F(2,3),F(1,3)],[F(1),F(-10)]]:
            rows.append(check(b,alpha,es,ks,ts,ws,K))
out=dict(label="E151",status="exact_three_cost_identity_and_parameter_recovery_checks",cases=rows,
    limitations="No noise robustness or empirical price/adoption claim. Finite prefix recovery in these examples does not establish uniform prefix length.")
(root/"switch_three_cost_common_filter_identification_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(dict(label="E151",passed=len(rows),nonzero_orders=sorted(set(x["first_nonzero_B"] for x in rows)),recovered=True)))
