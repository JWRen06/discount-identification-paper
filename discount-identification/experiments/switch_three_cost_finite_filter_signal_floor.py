from fractions import Fraction as F
from pathlib import Path
import json
root=Path(__file__).resolve().parent
src=(root/"switch_three_cost_common_filter_identification.py").read_text(encoding="utf-8").split("rows=[]")[0]
exec(compile(src,"E151_reused_definitions","exec"))
esA=[F(1,4),F(3,8),F(1,2)];esB=[F(2,7),F(4,9),F(3,5)]
ks=[F(1),F(2),F(15,4)];alpha=F(5,2)
assert all(alpha*e/((1-e)*(1-F(2,3)*e))==k for e,k in zip(esA,ks))
assert all(alpha*e/(1-e)==k for e,k in zip(esB,ks))
rows=[]
for N in [2,8,32]:
 for eps in [F(1,100),F(1,10000),F(1,10**8)]:
  diffs=[];balls=[]
  for ea,eb in zip(esA,esB):
   a=[eps*v for v in coeff(ea,F(2,3),[F(0)],[F(1)],N)]
   b=[eps*v for v in coeff(eb,F(0),[F(0)],[F(1)],N)]
   mid=[(x+y)/2 for x,y in zip(a,b)]
   da=sum(abs(x-y) for x,y in zip(mid,a));db=sum(abs(x-y) for x,y in zip(mid,b))
   assert da==db and da<=eps
   balls.append(str(da));diffs.append(str(2*da))
  rows.append(dict(N=N,epsilon=str(eps),filter_initial=str(eps),delayed_mass=str(1-eps),delay=N+1,
   prefix_distance_by_fee=diffs,midpoint_error_by_fee=balls,universal_beta_error_floor="1/3"))
out=dict(label="E155",status="exact_common_positive_unit_gain_filter_finite_noise_floor",cases=rows,
 limitations="Uniform finite-window class lower bound, not full-kernel nonidentification or fixed-filter asymptotic impossibility. Standard two-point construction.")
(root/"switch_three_cost_finite_filter_signal_floor_results.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(dict(label="E155",passed=len(rows),noise_ball_shared=True,beta_floor="1/3")))
