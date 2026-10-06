from fractions import Fraction as F
from pathlib import Path
import json
root=Path(__file__).resolve().parent
src=(root/"switch_three_cost_common_filter_identification.py").read_text(encoding="utf-8").split("rows=[]")[0]
exec(compile(src,"E151_definitions","exec"))
b=F(2,3);alpha=F(5,2);es=[F(1,4),F(3,8),F(1,2)]
ks=[F(1),F(2),F(15,4)]
mat=[[coeff(e,b,[s],[F(1)],0)[0] for s in ss] for e in es]+[[1-s for s in ss]]
a=[row[:] for row in mat];det=F(1)
for j in range(4):
 idx=next(i for i in range(j,4) if a[i][j]!=0)
 if idx!=j:a[j],a[idx]=a[idx],a[j];det=-det
 v=a[j][j];det*=v
 for i in range(j+1,4):
  ratio=a[i][j]/v
  for l in range(j,4):a[i][l]-=ratio*a[j][l]
assert det!=0 and all(w>0 for w in weights)
T=sum(w*(1-s) for w,s in zip(weights,ss))
identities=[]
for e,k in zip(es,ks):
 f=coeff(e,b,ss,weights,1)
 rhs=(1+1/b+alpha/(b*k))*f[0]-alpha*T/(b*k)
 assert f[1]==rhs
 identities.append(dict(fee=str(k),f0=str(f[0]),f1=str(f[1]),Euler_first_lag=True))
out=dict(label="E157",status="exact_positive_four_support_base_and_two_lag_identity",
 determinant_exact=str(det),targets=[str(s) for s in ss],weights=[str(w) for w in weights],T=str(T),
 baseline_filter=["1/2","1/4","1/4"],identity_checks=identities,
 proof_method="Nearby worlds defined by exact invertible linear solve and continuity, not numerical Jacobian. See H80 full proof.",
 limitations="Known alpha case not covered by this lower bound; no numerical perturbation advertised as exact world.")
(root/"switch_three_cost_two_lag_positive_base_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(dict(label="E157",determinant=str(det),positive_base=True,Euler_identities=3)))
