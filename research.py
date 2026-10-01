#!/usr/bin/env python3
"""VWAP research and depth-aware execution. Run from the extracted project:
python research.py --data data --out results
Set 1 alone selects the model. The exact frozen model is then replayed on set 2.
"""
from pathlib import Path
from decimal import Decimal as D
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

EXECUTION = dict(fee=.20, clip=50, position_limit=100, notional_limit=5000., buffer=.12)
LEVEL2 = [.05, .1, .25, .5, 1.]
LEVEL3 = [.0025, .01, .0625, .25, 1.]
TAUS = [10., 20., 30., 45., 60., 120.]
GAMMAS = [0., -.05, .05, -.1, .1, -.25, .25, -.5, .5]


def load(data, test=False):
    paths = []
    for kind in ['books', 'trades']:
        matches = [p for p in Path(data).glob('vwap_workshop_'+kind+'*.csv')
                   if ('_set2' in p.name) == test]
        if len(matches) != 1:
            raise ValueError(f'Expected one {kind} file for set {2 if test else 1}')
        paths.append(matches[0])
    book, trades = map(pd.read_csv, paths)
    assert len(book) == 20 and list(book.timestamp) == [f'T{i:02}' for i in range(1,21)]
    assert not book.isna().any().any() and not trades.isna().any().any()
    assert (pd.to_timedelta(book.time).diff().dropna().dt.total_seconds() == 5).all()
    for side in ['bid', 'ask']:
        px, qty, ages = arrays(book, side)
        assert (px > 0).all() and (qty > 0).all() and (ages >= 0).all()
        assert (qty == qty.astype(int)).all()
        assert (np.diff(px,axis=1) * (1 if side == 'ask' else -1) > 0).all()
    assert (book.ask1_px > book.bid1_px).all()
    assert set(trades.aggressor) <= {'BUY', 'SELL'}
    assert (trades.qty > 0).all() and (trades.qty == trades.qty.astype(int)).all()
    assert (trades.price > 0).all() and not trades.duplicated(['interval','seq']).any()
    for trade in trades.itertuples():
        start, end = trade.interval.split('->')
        j = list(book.timestamp).index(end)
        assert j > 0 and start == book.timestamp.iloc[j-1] and trade.printed_before == end
    book['mid'] = (book.bid1_px + book.ask1_px)/2
    book['spread'] = book.ask1_px - book.bid1_px
    trades['signed_qty'] = trades.qty * np.where(trades.aggressor == 'BUY', 1, -1)
    by_time = trades.groupby('printed_before')[['qty','signed_qty']].sum()
    book['flow'] = book.timestamp.map(by_time.signed_qty/by_time.qty).fillna(0.)
    book['trade_qty'] = book.timestamp.map(by_time.qty).fillna(0.)
    return book, trades, paths


def arrays(book, side):
    return [book[[f'{side}{i}_{field}' for i in range(1,4)]].to_numpy(float)
            for field in ['px', 'qty', 'age_s']]


def predict(book, model):
    if model['kind'] in ('mid','last_mid'):
        return book.mid.to_numpy()
    if model['kind'] == 'blend':
        base = predict(book, model['base'])
        return model.get('alpha', .5)*base + (1-model.get('alpha', .5))*book.mid.to_numpy()
    bp, bq, ba = arrays(book, 'bid')
    ap, aq, aa = arrays(book, 'ask')
    if model['kind'] == 'direct':
        return (aq*bp + bq*ap).sum(1)/(aq+bq).sum(1)
    weights = np.array([1., model['w2'], model['w3']])
    def decay(age):
        family = model.get('decay', 'none')
        x = age/model.get('tau', 30.)
        if family == 'none': return np.ones_like(x)
        if family == 'exp': return np.exp(-x)
        if family == 'gaussian': return np.exp(-x*x)
        if family == 'hyperbolic': return 1/(1+x)
        raise ValueError(f'Unknown decay: {family}')
    B, A = bq*weights*decay(ba), aq*weights*decay(aa)
    DB, DA = B.sum(1), A.sum(1)
    bid_vwap, ask_vwap = (B*bp).sum(1)/DB, (A*ap).sum(1)/DA
    theo = (DA*bid_vwap + DB*ask_vwap)/(DA+DB)
    return theo + model.get('gamma', 0.)*book.flow.to_numpy()*book.spread.to_numpy()


def metric(book, theo, idx=None):
    errors = theo[:-1] - book.mid.to_numpy()[1:]
    if idx is not None: errors = errors[idx]
    return dict(mae=float(np.abs(errors).mean()), rmse=float(np.sqrt(np.mean(errors**2))),
                bias=float(errors.mean()))


def walk(row, side, quantity, slip=0.):
    left = int(quantity)
    legs = []
    for level in range(1,4):
        size = min(left, int(row[f'{side}{level}_qty']))
        price = float(D(str(row[f'{side}{level}_px'])) + D(str(slip))*(1 if side=='ask' else -1))
        if size: legs.append((level,size,price))
        left -= size
        if left == 0: break
    return legs, left


def simulate(book, theo, fee=.20, clip=50, position_limit=100,
             notional_limit=5000., buffer=.12, slip=0., delay=0, fee_per_level=False, decision_fee=None):
    """One immediate order per snapshot. T20 is liquidation only, with no clip cap.
    Prices and cash movements use Decimal; theo/decision edges use floating point.
    Observed future books are exogenous: our fills do not alter subsequent snapshots.
    """
    if decision_fee is None: decision_fee=fee
    cash, paid_fees, position = D('0'), D('0'), 0
    decisions, fills = [], []
    order = 0
    for t, row in book.iterrows():
        signal = float(theo[max(0,t-delay)])
        action, direction, legs, residual = 'HOLD', 0, [], 0
        best = 0.
        edges = {'BUY': None, 'SELL': None}
        reason = 'No positive executable surplus'
        if t == len(book)-1:
            reason = 'Scheduled end-of-replay liquidation'
            if position:
                direction = -1 if position > 0 else 1
                legs, residual = walk(row, 'bid' if direction<0 else 'ask', abs(position), slip)
                action = 'LIQUIDATE'
        elif t >= delay:
            for sign, side, label in [(1,'ask','BUY'),(-1,'bid','SELL')]:
                for size in range(1,clip+1):
                    after = position + sign*size
                    if abs(after)>position_limit or abs(after)*row.mid>notional_limit: continue
                    candidate, remaining = walk(row,side,size,slip)
                    if remaining: continue
                    charge = decision_fee*(len(candidate) if fee_per_level else 1)
                    surplus = sum(q*sign*(signal-p) for _,q,p in candidate)-charge-buffer*row.spread*size
                    edges[label] = max(edges[label] if edges[label] is not None else -np.inf, surplus)
                    # Exact surplus ties keep the first candidate (smaller size).
                    if surplus > best+1e-12:
                        best, direction, legs, action = surplus, sign, candidate, label
                        reason = 'Maximum positive executable surplus'
        step_fee = D(str(fee))*(len(legs) if fee_per_level else 1) if legs else D('0')
        quantity = sum(q for _,q,_ in legs)
        if legs:
            order += 1
            for level,size,price in legs:
                change = -direction*size*D(str(price))
                cash += change
                fills.append(dict(timestamp=row.timestamp,order=order,action=action,
                                  side='BUY' if direction>0 else 'SELL',level=level,qty=size,
                                  price=price,signed_qty=direction*size,cash_delta=float(change)))
            position += direction*quantity
            cash -= step_fee
            paid_fees += step_fee
        equity = cash + position*D(str(row.mid))
        remaining_book = row.copy()
        for level,size,_ in legs:
            remaining_book[f'{"ask" if direction>0 else "bid"}{level}_qty'] -= size
        exit_legs, missing = walk(remaining_book,'bid' if position>0 else 'ask',abs(position),slip)
        liquidation = cash
        if position:
            liquidation = (cash + (1 if position>0 else -1)*sum(D(str(p))*q for _,q,p in exit_legs)
                           - D(str(fee))*(len(exit_legs) if fee_per_level else 1)) if not missing else None
        decisions.append(dict(timestamp=row.timestamp,time=row.time,mid=row.mid,theo=signal,
                              best_bid=row.bid1_px,best_ask=row.ask1_px,flow=row.flow,
                              buy_surplus=edges['BUY'],sell_surplus=edges['SELL'],action=action,
                              qty=quantity,avg_price=sum(q*p for _,q,p in legs)/quantity if quantity else None,
                              fee=float(step_fee),cum_fees=float(paid_fees),position=position,cash=float(cash),
                              equity=float(equity),liquidation_value=float(liquidation) if liquidation is not None else None,
                              notional=abs(position)*row.mid,surplus=best,unfilled=residual,reason=reason))
    ledger = pd.DataFrame(decisions)
    fills = pd.DataFrame(fills,columns=['timestamp','order','action','side','level','qty','price','signed_qty','cash_delta'])
    curve = np.r_[0.,ledger.equity.to_numpy()]
    summary = dict(net_pnl=float(cash) if position==0 else None,
                   gross_pnl=float(cash+paid_fees) if position==0 else None,fees=float(paid_fees),
                   orders=order,filled_units=int(fills.qty.sum()),terminal_position=position,
                   peak_notional=float(ledger.notional.max()),max_drawdown=float((np.maximum.accumulate(curve)-curve).max()))
    assert np.isclose(float(cash),fills.cash_delta.sum()-float(paid_fees))
    assert position == fills.signed_qty.sum()
    return ledger, fills, summary


def calibrate(book):
    """Full set 1 net P&L objective. MAE is diagnostic, not a second objective."""
    stages, history = [], []
    def score(model):
        theo = predict(book,model)
        result = simulate(book,theo,**EXECUTION)[2]
        assert result['terminal_position']==0
        return {**result,**metric(book,theo)}
    def rank(row): return (-round(row['net_pnl'],9),row['orders'])
    def add(name,model,note): stages.append(dict(name=name,model=model,note=note,**score(model)))
    def tune(stage,candidates):
        scores=[]
        for model in candidates:
            row=dict(stage=stage,params=json.dumps(model,sort_keys=True),**score(model))
            history.append(row);scores.append((row,model))
        return min(scores,key=lambda pair:rank(pair[0]))[1]
    add('Midpoint',{'kind':'mid'},'No-change benchmark; cannot clear the spread.')
    add('VWAP 1',{'kind':'direct'},'Direct crossed depth reference.')
    base=dict(kind='weighted',w2=.25,w3=.0625,decay='none')
    add('VWAP 2 fixed',base,'Fixed geometric depth weights.')
    add('VWAP 3 fixed',{**base,'decay':'exp','tau':30.},'Original age-decay reference.')
    geom=tune('Geometric weights',[base]+[{**base,'w2':x,'w3':x*x} for x in LEVEL2 if x!=.25])
    add('Geometric tuned',geom,'Choose geometric depth weights by training net P&L.')
    indep=tune('Independent weights',[geom]+[{**base,'w2':u,'w3':v} for u in LEVEL2 for v in LEVEL3 if v<=u])
    add('Free weights (1,w1,w2)',indep,'Allow independent non-increasing level weights (1,w1,w2).')
    age=tune('Age decay',[indep]+[{**indep,'decay':d,'tau':tau} for d in ['exp','gaussian','hyperbolic'] for tau in TAUS])
    add('Age tuned',age,'Test age decay, retaining no decay as the first candidate.')
    flow=tune('Trade flow',[{**age,'gamma':g} for g in GAMMAS])
    add('Flow tuned',flow,'Test last-completed-interval flow, retaining zero coefficient.')
    return stages,pd.DataFrame(history),min(stages,key=rank)


def audit(book, ledger, fills, summary):
    """Independent FIFO matching and per-timestamp depth/inventory checks."""
    from collections import deque
    lots=deque(); gross=D('0'); inventory=0; cash=D('0')
    for row in ledger.itertuples():
        now=fills[fills.timestamp==row.timestamp]
        source=book[book.timestamp==row.timestamp].iloc[0]
        for (side,level), group in now.groupby(['side','level']):
            assert group.qty.sum() <= source[f'{"ask" if side=="BUY" else "bid"}{level}_qty']
        for f in now.itertuples():
            sign=1 if f.side=='BUY' else -1
            size=int(f.qty); price=D(str(f.price))
            assert price == D(str(source[f'{"ask" if sign>0 else "bid"}{f.level}_px']))
            cash-=sign*size*price;inventory+=sign*size
            while size and lots and lots[0][0]!=sign:
                old_sign,old_size,old_price=lots.popleft()
                matched=min(size,old_size)
                gross+=old_sign*matched*(price-old_price)
                size-=matched;old_size-=matched
                if old_size:lots.appendleft((old_sign,old_size,old_price))
            if size:lots.append((sign,size,price))
        cash-=D(str(row.fee))
        assert inventory==row.position and abs(inventory)<=EXECUTION['position_limit']
        assert row.notional<=EXECUTION['notional_limit']
        assert np.isclose(float(cash),row.cash,atol=1e-9)
        assert np.isclose(row.equity,row.cash+row.position*row.mid,atol=1e-9)
    assert not lots and inventory==0
    assert gross-D(str(summary['fees']))==cash
    assert np.isclose(float(cash),summary['net_pnl'],atol=1e-9)
    return dict(fifo_gross=float(gross),fees=summary['fees'],net_cash=float(cash),terminal_position=inventory)


def extensions(book, selected_model):
    """Post-selection diagnostics. These do not alter the frozen strategy."""
    model_candidates={
      'Last midpoint': {'kind':'last_mid'},
      'Selected book+flow': selected_model,
    }
    rows=[]
    base=selected_model
    for alpha in [0,.25,.5,.75,1.0]:
        model_candidates[f'Blend alpha={alpha:g}']={'kind':'blend','base':base,'alpha':alpha}
    for name,model in model_candidates.items():
        pr=predict(book,model)
        rows.append(dict(model=name,**metric(book,pr),**simulate(book,pr,**EXECUTION)[2]))
    # Free weights explicitly, with first level fixed at one.
    for w1 in [.05,.1,.25,.5,1.0]:
      for w2 in [.0025,.01,.0625,.25,1.0]:
        if w2<=w1:
          model={'kind':'weighted','w2':w1,'w3':w2,'decay':'none'}
          rows.append(dict(model=f'Weights (1,{w1:g},{w2:g})',**metric(book,predict(book,model)),**simulate(book,predict(book,model),**EXECUTION)[2]))
    # Trading rule sensitivity. Score fees and actual P&L fees are deliberately separated.
    policies=[]
    for label,kw in [('Base',{}),('No fee in G',{'decision_fee':0}),('No spread buffer',{'buffer':0}),('Buffer 0.06',{'buffer':.06}),('Buffer 0.24',{'buffer':.24}),('Decision fee 0.40',{'decision_fee':.4})]:
      policies.append((label,kw))
    for label,kw in policies:
      rows.append(dict(model='Policy: '+label,**metric(book,predict(book,base)),**simulate(book,predict(book,base),**{**EXECUTION,**kw})[2]))
    # Maximum feasible quantity: remove clip cap but retain notional and position controls.
    for clip in [50,100,250,1000]:
      rows.append(dict(model=f'Max quantity clip={clip}',**metric(book,predict(book,base)),**simulate(book,predict(book,base),**{**EXECUTION,'clip':clip})[2]))
    return pd.DataFrame(rows)

def run(data, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    b1,tr1,p1=load(data)
    stages,search,selected=calibrate(b1)
    frozen=dict(objective='Maximum full set 1 net P&L',selected_stage=selected['name'],model=selected['model'],
                execution=EXECUTION,tie_rule='Fewer orders, then original candidate order',
                candidate_count=len(search),note='Training-fit result, not cross-validated performance.')
    (out/'frozen_config.json').write_text(json.dumps(frozen,indent=2))
    # No test data enters the calibration or freezing functions above.
    b2,tr2,p2=load(data,True)
    books={'Set 1':b1,'Set 2':b2};trades={'Set 1':tr1,'Set 2':tr2};replays={};audits={}
    for label,book in books.items():
        theo=predict(book,selected['model'])
        replay=simulate(book,theo,**EXECUTION)
        replays[label]=replay;audits[label]=audit(book,*replay)
        slug=label.lower().replace(' ','_')
        pred=book[['timestamp','time','mid','flow']].copy();pred['next_mid']=book.mid.shift(-1)
        for stage in stages:pred[stage['name']]=predict(book,stage['model'])
        pred['selected_theo']=theo;pred.to_csv(out/f'{slug}_predictions.csv',index=False)
        replay[0].to_csv(out/f'{slug}_decisions.csv',index=False)
        replay[1].to_csv(out/f'{slug}_fills.csv',index=False)
    training=pd.DataFrame([{k:v for k,v in stage.items() if k!='model'} for stage in stages])
    training.to_csv(out/'training_stages.csv',index=False);search.to_csv(out/'training_search.csv',index=False)
    sensitivity=[]
    for label,book in books.items():
        for scenario,extra in [('Base',{}),('Fee per level',{'fee_per_level':True}),('0.5p extra each side',{'slip':.005}),('One snapshot stale theo',{'delay':1}),('Double fee',{'fee':.4})]:
            sm=simulate(book,predict(book,selected['model']),**{**EXECUTION,**extra})[2]
            sensitivity.append(dict(dataset=label,scenario=scenario,**sm))
    sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(out/'sensitivity.csv',index=False)
    ext=pd.concat([extensions(book,selected['model']).assign(dataset=label) for label,book in books.items()],ignore_index=True)
    ext.to_csv(out/'extensions.csv',index=False)
    manifest={**frozen,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in p1+p2},
              'results':{label:rep[2] for label,rep in replays.items()},'independent_fifo_audits':audits}
    (out/'audit.json').write_text(json.dumps(manifest,indent=2))
    from report_builder import build
    build(out,books,trades,stages,search,selected,replays,sensitivity,manifest,ext)
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data');parser.add_argument('--out',default='results')
    args=parser.parse_args();print(json.dumps(run(args.data,args.out),indent=2))
