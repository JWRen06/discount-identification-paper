"""Read-only, exact-decimal audit of issuer-published holdings; no preference fitting."""
from pathlib import Path
from decimal import Decimal, getcontext
import csv, io, json, hashlib, re
from collections import Counter, defaultdict
getcontext().prec = 50
ROOT = Path(__file__).resolve().parent
raw = (ROOT / 'MTUM_latest.csv').read_bytes()
lines = raw.decode('utf-8-sig').splitlines()
start = next(i for i,x in enumerate(lines) if x.startswith('Ticker,Name,'))
meta = dict((r[0],r[1]) for r in csv.reader(lines[:start]) if len(r)==2)
rows = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
page=(ROOT/'ishares_product.html').read_text(encoding='utf-8')
linked=[json.loads(x) for x in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',page,re.S)]
def props(x):
    if isinstance(x,dict):
        if x.get('name')=='Net Assets of Fund' and 'value' in x: yield x
        for v in x.values(): yield from props(v)
    elif isinstance(x,list):
        for v in x: yield from props(v)
nav_fields=list(props(linked))
assert len(nav_fields)==1
nav_field=nav_fields[0]
assert nav_field['valueReference']['value']==meta['Fund Holdings as of']
assert rows and all(len(r)==15 and None not in r for r in rows)
D = lambda s: Decimal(s.replace(',',''))
def val(r,k): return D(r[k])
mv = sum((val(r,'Market Value') for r in rows),Decimal(0))
weight = sum((val(r,'Weight (%)') for r in rows),Decimal(0))
official_nav=D(nav_field['value'].replace('$',''))
eq = [r for r in rows if r['Asset Class']=='Equity']
assert eq and all(r['Currency']=='USD' and r['FX Rate']=='1.00' for r in eq)
assert all(val(r,'Quantity')>0 and val(r,'Market Value')>0 for r in eq)
counts = Counter(r['Ticker'] for r in rows)
out, lows, highs = [],[],[]
for r in rows:
    v,q,p,w = [val(r,k) for k in ('Market Value','Quantity','Price','Weight (%)')]
    reconstructed_weight = 100*v/mv
    # Weight is published to 0.01 percentage point. Common denominator
    # feasibility is a rounding check, not an independent NAV estimate.
    if v>0:
        lows.append(100*v/(w+Decimal('.005')))
        if w>Decimal('.005'): highs.append(100*v/(w-Decimal('.005')))
    residual = v-q*p
    out.append(dict(ticker=r['Ticker'],asset_class=r['Asset Class'],
                    value_usd=str(v),quantity=str(q),price=str(p),
                    reported_weight_pct=str(w),weight_from_sum_values_pct=str(reconstructed_weight),
                    weight_difference_ppt=str(reconstructed_weight-w),
                    naive_quantity_times_price_residual_usd=str(residual),
                    equity_identity_checked=r['Asset Class']=='Equity'))
sectors=defaultdict(lambda: Decimal(0))
for r in eq: sectors[r['Sector']]+=val(r,'Market Value')
top=sorted(eq,key=lambda r:val(r,'Market Value'),reverse=True)
result = dict(source='https://www.ishares.com/us/products/251614/ishares-msci-usa-momentum-factor-etf/latest-holdings.csv',
              reported_date=meta['Fund Holdings as of'],shares_outstanding=meta['Shares Outstanding'],
              original_sha256=hashlib.sha256(raw).hexdigest(),records=len(rows),equity_records=len(eq),
              product_page_sha256=hashlib.sha256((ROOT/'ishares_product.html').read_bytes()).hexdigest(),
              same_date_official_nav_usd=str(official_nav),
              holdings_value_minus_nav_usd=str(mv-official_nav),
              holdings_value_minus_nav_pct=str(100*(mv/official_nav-1)),
              weights_failing_rounding_if_official_nav_used=sum(abs(100*val(r,'Market Value')/official_nav-val(r,'Weight (%)'))>Decimal('.005') for r in rows),
              asset_class_counts=dict(Counter(r['Asset Class'] for r in rows)),
              duplicate_tickers={k:v for k,v in counts.items() if v>1},
              sum_reported_values_usd=str(mv),sum_reported_weights_pct=str(weight),
              max_equity_value_minus_quantity_price_usd=str(max(abs(val(r,'Market Value')-val(r,'Quantity')*val(r,'Price')) for r in eq)),
              max_weight_difference_ppt=str(max(abs(D(x['weight_difference_ppt'])) for x in out)),
              weights_within_half_rounding_unit=sum(abs(D(x['weight_difference_ppt']))<=Decimal('.005') for x in out),
              feasible_common_denominator_usd=[str(max(lows)),str(min(highs))],
              sum_values_in_denominator_interval=max(lows)<=mv<=min(highs),
              top10_value_share_pct=str(100*sum((val(r,'Market Value') for r in top[:10]),Decimal(0))/mv),
              equity_sector_value_share_pct={k:str(100*v/mv) for k,v in sorted(sectors.items(),key=lambda x:x[1],reverse=True)},
              non_equity_records=[x for x in out if not x['equity_identity_checked']],
              top10=[dict(ticker=r['Ticker'],reported_weight_pct=r['Weight (%)']) for r in top[:10]])
assert result['max_equity_value_minus_quantity_price_usd']=='0.0000'
assert result['weights_within_half_rounding_unit']==len(rows)
assert result['sum_values_in_denominator_interval'] and not result['duplicate_tickers']
(ROOT/'holdings_audit_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'holdings_row_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(out[0]));wr.writeheader();wr.writerows(out)
print(json.dumps(result,ensure_ascii=False,indent=2))
