from pathlib import Path
from fractions import Fraction as F
import json
root=Path(__file__).resolve().parent
src=(root/"switch_three_cost_common_filter_identification.py").read_text(encoding="utf-8").split("rows=[]")[0]
exec(compile(src,"E151_reused_definitions","exec"))
def norm(a):return sum(abs(v) for v in a)
b=F(2,3);alpha=F(5,2);es=[F(1,4),F(3,8),F(1,2)];ks=[F(1),F(2),F(15,4)];K=[F(2,3),F(1,3)]
N=2;trueZ=[conv(K,coeff(e,b,ss,weights,N),N) for e in es]
rows=[]
for eps in [F(1,10**8),F(1,10**6),F(1,10**4)]:
 Z=[[v+eps*F(1 if (n+j)%2==0 else -1,3) for n,v in enumerate(z)] for j,z in enumerate(trueZ)]
 assert all(norm(add(zz,tz,F(1),F(-1)))==eps for zz,tz in zip(Z,trueZ))
 c=[k*z[0] for k,z in zip(ks,Z)];dc=c[2]-c[0];db=c[1]-c[0];km=max(ks)
 V=add(Z[1],Z[0],F(1),F(-1));W=add(Z[2],Z[0],F(1),F(-1))
 VK=add(Z[1],Z[0],ks[1],-ks[0]);WK=add(Z[2],Z[0],ks[2],-ks[0])
 A=add(V,W,dc,-db);B=add(VK,WK,dc,-db)
 ae=2*eps*(abs(dc)+abs(db))+2*km*eps*(norm(V)+norm(W))+8*km*eps**2
 be=2*km*eps*(abs(dc)+abs(db))+2*km*eps*(norm(VK)+norm(WK))+8*km**2*eps**2
 u=conv([F(1),F(-1)],B,N)
 M=[[A[0],-u[1]],[A[1],-u[2]]];rhs=[-u[0],-u[1]]
 det=M[0][0]*M[1][1]-M[0][1]*M[1][0]
 inv=[[M[1][1]/det,-M[0][1]/det],[-M[1][0]/det,M[0][0]/det]]
 th=[sum(v*w for v,w in zip(row,rhs)) for row in inv]
 q=max(norm(row) for row in inv);eta=ae+2*be;rho=2*be
 row=dict(epsilon=str(eps),alpha_estimate=str(th[0]),beta_estimate=str(th[1]),q_eta=str(q*eta),q_eta_display=float(q*eta),
   actual_error_display=float(max(abs(th[0]-alpha),abs(th[1]-b))))
 if q*eta<1:
  rad=q*(rho+eta*max(abs(x) for x in th))/(1-q*eta)
  assert max(abs(th[0]-alpha),abs(th[1]-b))<=rad
  row.update(status="certified",radius_exact=str(rad),radius_display=float(rad))
 else:row.update(status="sufficient_certificate_uninformative")
 rows.append(row)
out=dict(label="E152",status="exact_finite_noise_certificate_checks",cases=rows,rows_used=[1,2],last_coefficient=2,
 limitation="Synthetic bounded coefficient error and model-shared kernel; no sampling-cost or market-price validity. Sufficient perturbation bound, not optimal risk.")
(root/"switch_three_cost_finite_noise_certificate_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(dict(label="E152",cases=[{k:v for k,v in x.items() if k in ["epsilon","status","radius_display","actual_error_display","q_eta_display"]} for x in rows])))
