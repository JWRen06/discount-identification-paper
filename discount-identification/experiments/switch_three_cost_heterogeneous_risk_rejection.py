from fractions import Fraction as F
from pathlib import Path
import json
root=Path(__file__).resolve().parent
src=(root/"switch_three_cost_common_filter_identification.py").read_text(encoding="utf-8").split("rows=[]")[0]
exec(compile(src,"E151_definitions","exec"))
ks=[F(1),F(2),F(4)];N=4
Z=[add(coeff(k/(k+1),F(0),[F(0)],[F(1)],N),coeff(k/(k+4),F(0),[F(0)],[F(1)],N)) for k in ks]
c=[k*z[0] for k,z in zip(ks,Z)];dc=c[2]-c[0];db=c[1]-c[0]
V=add(Z[1],Z[0],F(1),F(-1));W=add(Z[2],Z[0],F(1),F(-1))
VK=add(Z[1],Z[0],ks[1],-ks[0]);WK=add(Z[2],Z[0],ks[2],-ks[0])
A=add(V,W,dc,-db);B=add(VK,WK,dc,-db);u=conv([F(1),F(-1)],B,N)
delta=A[0]*(B[2]-B[1])-A[1]*B[1];ah=B[1]**2/delta;bh=A[0]*B[1]/delta
assert ah==F(22898,9125) and bh==F(963,1825)
res=[ah*A[n-1]-bh*u[n]+u[n-1] for n in [3,4]]
assert res==[F(3759,91250),F(40131,912500)]
def norm(a):return sum(abs(v) for v in a)
M=[[A[0],-u[1]],[A[1],-u[2]]];det=M[0][0]*M[1][1]-M[0][1]*M[1][0]
inv=[[M[1][1]/det,-M[0][1]/det],[-M[1][0]/det,M[0][0]/det]]
q=max(norm(row) for row in inv)
rows=[]
for eps in [F(1,10**8),F(1,10**6),F(1,10**4)]:
 km=max(ks)
 ae=2*eps*(abs(dc)+abs(db))+2*km*eps*(norm(V)+norm(W))+8*km*eps**2
 be=2*km*eps*(abs(dc)+abs(db))+2*km*eps*(norm(VK)+norm(WK))+8*km**2*eps**2
 eta=ae+2*be;rho=2*be
 row=dict(epsilon=str(eps),q_eta=str(q*eta),q_eta_display=float(q*eta))
 if q*eta<1:
  rad=q*(rho+eta*max(abs(ah),abs(bh)))/(1-q*eta)
  cap=rad*(abs(A[2])+abs(u[3]))+(abs(ah)+rad)*ae+(abs(bh)+rad)*2*be+2*be
  row.update(radius_display=float(rad),residual_cap_exact=str(cap),residual_cap_display=float(cap),rejected=abs(res[0])>cap)
 else:row.update(status="parameter_certificate_uninformative",rejected=False)
 rows.append(row)
out=dict(label="E156",status="exact_heterogeneous_risk_spurious_discount_and_extra_coefficient_rejection",
 true_common_beta="0",risk_types=["1","4"],apparent_alpha=str(ah),apparent_beta=str(bh),residuals=[str(v) for v in res],noise_checks=rows,
 limitations="Plausible parameter output is not an observationally equivalent common positive world; necessary residual rejection only, not identification of failure cause.")
(root/"switch_three_cost_heterogeneous_risk_rejection_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(dict(label="E156",apparent_beta=str(bh),residuals=[str(v) for v in res],noise_checks=[{k:v for k,v in x.items() if k in ["epsilon","rejected","status","residual_cap_display"]} for x in rows])))
