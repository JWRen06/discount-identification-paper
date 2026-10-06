from fractions import Fraction as F
from pathlib import Path
import json
root=Path(__file__).resolve().parent
src=(root/"switch_three_cost_common_filter_identification.py").read_text(encoding="utf-8").split("rows=[]")[0]
exec(compile(src,"E151_reused_definitions","exec"))
rows=[]
for b in [F(0),F(1,10),F(2,3),F(99,100)]:
 alpha=F(5,2);es=[F(1,4),F(3,8),F(1,2)]
 ks=[alpha*e/((1-e)*(1-b*e)) for e in es]
 for ts,ws in [(ss,weights),([F(0)],[F(1)]),([F(1,10),F(99,100),F(1)],[F(1,100),F(1),F(5)])]:
  for K in [[F(1)],[F(2,3),F(1,3)],[F(-1),F(10)]]:
   Z=[conv(K,coeff(e,b,ts,ws,2),2) for e in es]
   c=[k*z[0] for k,z in zip(ks,Z)]
   A=add(add(Z[1],Z[0],F(1),F(-1)),add(Z[2],Z[0],F(1),F(-1)),c[2]-c[0],-(c[1]-c[0]))
   B=add(add(Z[1],Z[0],ks[1],-ks[0]),add(Z[2],Z[0],ks[2],-ks[0]),c[2]-c[0],-(c[1]-c[0]))
   delta=A[0]*(B[2]-B[1])-A[1]*B[1]
   assert B[0]==0 and B[1]<0 and delta>0
   assert B[1]**2/delta==alpha and A[0]*B[1]/delta==b
   assert (A[0]==0) if b==0 else (A[0]<0)
   rows.append(dict(beta=str(b),alpha=str(alpha),K=[str(v) for v in K],B1=str(B[1]),Delta=str(delta),beta_recovered=str(A[0]*B[1]/delta),alpha_recovered=str(B[1]**2/delta)))
out=dict(label="E153",status="exact_three_coefficient_universal_formula_checks",cases=rows,
 limitations="Finite examples validate algebra, not replace strict concavity proof. Nonzero K0 shared across fees; no shortest time-horizon proof or noise optimality.")
(root/"switch_three_cost_three_lag_positive_pool_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(dict(label="E153",passed=len(rows),all_three_coefficient_formulas=True)))
