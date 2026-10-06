from decimal import Decimal, getcontext
from pathlib import Path
import json
getcontext().prec=45
base=Path(__file__).resolve().parent
f=json.loads((base/'mtum_two_snapshot_fields_20261006.json').read_text(encoding='utf-8'))
D=Decimal
a,b=f['snapshots']
navratio=D(b['net_assets_usd'])/D(a['net_assets_usd'])
result={'observation_id':'M01','net_asset_growth_percent':str((navratio-1)*100),'holdings':{},'flows':[]}
for symbol in a['holdings']:
    x,y=a['holdings'][symbol],b['holdings'][symbol]
    assert x['cusip']==y['cusip']
    qratio=D(y['shares'])/D(x['shares'])
    vratio=D(y['value_usd'])/D(x['value_usd'])
    pratio=vratio/qratio
    weights=[100*D(z['value_usd'])/D(s['net_assets_usd']) for z,s in [(x,a),(y,b)]]
    errors=[abs(w-D(z['weight_percent'])) for w,z in zip(weights,[x,y])]
    assert max(errors)<D('0.000000000001')
    residual=abs(weights[1]/weights[0]-qratio*pratio/navratio)
    assert residual<D('1e-40')
    result['holdings'][symbol]={'reported_balance_change':str(D(y['shares'])-D(x['shares'])),'reported_balance_growth_percent':str((qratio-1)*100),'value_growth_percent':str((vratio-1)*100),'inferred_unit_value_growth_percent':str((pratio-1)*100),'weight_change_percentage_points':str(D(y['weight_percent'])-D(x['weight_percent'])),'weight_relative_change_percent':str((weights[1]/weights[0]-1)*100),'reported_weight_error_percent':list(map(str,errors)),'decomposition_residual':str(residual),'direction_reversal':qratio>1 and weights[1]<weights[0]}
for s in [a,b]:
    sold=sum(map(D,s['preceding_three_months_sold_usd']))
    redeemed=sum(map(D,s['preceding_three_months_redeemed_usd']))
    result['flows'].append({'snapshot_date':s['date'],'sold_usd':str(sold),'redeemed_usd':str(redeemed),'net_usd':str(sold-redeemed),'window':'preceding three months; windows do not cover entire endpoint interval'})
(base/'mtum_two_snapshot_results_20261006.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
