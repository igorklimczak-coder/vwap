"""Report presentation only. Numerical results come exclusively from research.py."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from research import predict, metric, EXECUTION, LEVEL2, LEVEL3, TAUS, GAMMAS

COLORS={'Midpoint':'#8493a5','VWAP 1':'#3671aa','VWAP 2 fixed':'#bf7736','VWAP 3 fixed':'#9275af',
        'Geometric tuned':'#af9871','Independent tuned':'#508a81','Age tuned':'#798167','Flow tuned':'#09655e'}

def money(x): return ('−' if x<0 else '')+f'£{abs(x):,.2f}'
def esc(x): return html.escape(str(x))
def math(tex, caption=''):
    return '<div class="formula"><div class="math" data-tex="'+html.escape(tex,quote=True)+'"></div>'+('<p>'+caption+'</p>' if caption else '')+'</div>'
def table(frame, digits=3):
    if frame.empty: return '<p class="muted">No executions.</p>'
    return '<div class="table-wrap">'+frame.to_html(index=False,border=0,na_rep='—',float_format=lambda x:f'{x:,.{digits}f}')+'</div>'
def details(label,body): return '<details><summary>'+label+'</summary>'+body+'</details>'
def plot(fig, height=430, ytitle='Price (£)', top=70, bottom=50):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=57,r=24,t=top,b=bottom),
                      font=dict(family='Arial, sans-serif',size=12,color='#33465a'),
                      legend=dict(orientation='h',y=1.02,yanchor='bottom',x=0,font=dict(size=11)),
                      hovermode='x unified',paper_bgcolor='white',plot_bgcolor='white')
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor='#edf0f4',zeroline=False)
    if ytitle:fig.update_yaxes(title_text=ytitle)
    return fig.to_html(full_html=False,include_plotlyjs=False,config=dict(responsive=True,displaylogo=False,scrollZoom=False))

def book_timeline(b):
    fig=go.Figure()
    for side,color in [('bid','#188378'),('ask','#cf7962')]:
        for level in range(1,4):
            fig.add_scatter(x=b.timestamp,y=b[f'{side}{level}_px'],name=f'{side.title()} {level}',mode='markers',
                            marker=dict(color=color,size=b[f'{side}{level}_qty'],sizemode='area',sizeref=.85,sizemin=3,opacity=.50),
                            customdata=np.c_[b[f'{side}{level}_qty'],b[f'{side}{level}_age_s']],
                            hovertemplate='%{x}<br>£%{y:.3f}<br>Size %{customdata[0]}<br>Age %{customdata[1]}s<extra>%{fullData.name}</extra>')
    fig.add_scatter(x=b.timestamp,y=b.mid,name='Midpoint',line=dict(color='#263e53',width=2))
    fig.update_layout(yaxis=dict(range=[8.27,8.83]))
    return plot(fig,410)

def ladder(b):
    def traces(t):
        r=b.iloc[t];ds=[]
        for side,sign,color in [('bid',-1,'#188378'),('ask',1,'#cf7962')]:
            ds.append(go.Bar(x=[sign*r[f'{side}{i}_qty'] for i in range(1,4)],y=[r[f'{side}{i}_px'] for i in range(1,4)],
                             width=.018,orientation='h',name=side.title(),marker_color=color,
                             customdata=[[r[f'{side}{i}_qty'],r[f'{side}{i}_age_s']] for i in range(1,4)],
                             hovertemplate='£%{y:.3f}<br>Size %{customdata[0]}<br>Age %{customdata[1]}s<extra>%{fullData.name}</extra>'))
        ds.append(go.Scatter(x=[0],y=[r.mid],name='Midpoint',mode='markers',marker=dict(color='#263e53',size=11,symbol='diamond')))
        return ds
    fig=go.Figure(data=traces(0),frames=[go.Frame(data=traces(i),name=str(i)) for i in range(20)])
    cap=float(b[[f'{s}{i}_qty' for s in ['bid','ask'] for i in range(1,4)]].max().max())*1.1
    fig.update_layout(barmode='overlay',yaxis=dict(range=[8.27,8.83]),xaxis=dict(range=[-cap,cap],title='Displayed size · bids left, asks right'),
                      sliders=[dict(y=-.26,currentvalue=dict(prefix='Snapshot '),steps=[dict(label=b.timestamp.iloc[i],method='animate',args=[[str(i)],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i in range(20)])])
    return plot(fig,460,top=45,bottom=125)

def result_chart(b, selected, stages, fills):
    fig=go.Figure()
    fig.add_scatter(x=b.timestamp,y=b.bid1_px,name='Best bid',line=dict(color='#c5ccd4',width=1))
    fig.add_scatter(x=b.timestamp,y=b.ask1_px,name='Best ask',line=dict(color='#c5ccd4',width=1),fill='tonexty',fillcolor='rgba(160,175,190,.10)')
    fig.add_scatter(x=b.timestamp,y=b.mid,name='Midpoint',line=dict(color='#8392a4',dash='dot'))
    for stage in stages:
        if stage['name'] in ['Midpoint',selected['name']]:continue
        fig.add_scatter(x=b.timestamp,y=predict(b,stage['model']),name=stage['name'],visible='legendonly',line=dict(color=COLORS[stage['name']]))
    fig.add_scatter(x=b.timestamp,y=predict(b,selected['model']),name='Selected theo',line=dict(color='#09655e',width=3),mode='lines+markers',marker_size=5)
    for side,symbol,color in [('BUY','triangle-up','#188378'),('SELL','triangle-down','#c36c59')]:
        ff=fills[fills.side==side]
        fig.add_scatter(x=ff.timestamp,y=ff.price,name=side,mode='markers',marker=dict(symbol=symbol,size=12,color=color),
                        customdata=np.c_[ff.qty,ff.level],hovertemplate='%{x}<br>£%{y:.3f}<br>%{customdata[0]} units · level %{customdata[1]}<extra>%{fullData.name}</extra>')
    return plot(fig,490,top=110)

def pnl_chart(ledger):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.12,row_heights=[.65,.35])
    for field,name,color in [('equity','Mid-marked P&L','#09655e'),('liquidation_value','Executable liquidation value','#c18b45')]:
        fig.add_scatter(x=ledger.timestamp,y=ledger[field],name=name,line=dict(color=color,width=2),row=1,col=1)
    fig.add_bar(x=ledger.timestamp,y=ledger.position,name='Position',marker_color='#a5b9ca',row=2,col=1)
    fig.update_yaxes(title_text='P&L (£)',row=1,col=1);fig.update_yaxes(title_text='Units',row=2,col=1)
    return plot(fig,450,ytitle=None)

CSS='''
:root{--ink:#20354a;--muted:#687888;--line:#dfe6eb;--teal:#09655e;--paper:#fff;--bg:#f5f7f9}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.65 Arial,sans-serif}
aside{position:fixed;left:0;top:0;bottom:0;width:210px;background:#183347;padding:35px 23px;color:#fff}aside .brand{font-size:11px;letter-spacing:2px;color:#92c8c2;margin-bottom:9px}aside h2{font-size:24px;line-height:1.25;margin:0 0 35px}aside a{display:block;color:#c6d4df;text-decoration:none;padding:10px 0;font-size:13px}aside a:hover{color:#fff}aside small{display:block;color:#91a7b9;margin-top:25px;line-height:1.6}
main{margin-left:210px;max-width:1320px;padding:36px 48px 50px}.masthead{margin-bottom:28px}.eyebrow{font-size:11px;letter-spacing:2px;color:var(--teal);font-weight:bold}h1{font-size:36px;letter-spacing:-1px;line-height:1.15;margin:12px 0}h2{font-size:25px;line-height:1.25;margin:0 0 13px;letter-spacing:-.3px}h3{font-size:18px;margin:24px 0 8px}p{margin:10px 0}.intro{font-size:17px;color:var(--muted);max-width:790px}.section{background:white;border:1px solid var(--line);border-radius:8px;padding:28px;margin-bottom:24px}.kicker{font-size:11px;color:var(--teal);font-weight:bold;letter-spacing:1.6px;margin-bottom:7px}.muted,.caption{color:var(--muted)}.caption{font-size:12px}.pill{display:inline-block;font-size:11px;background:#e7f1f0;color:var(--teal);border-radius:4px;padding:3px 8px}.row{display:grid;grid-template-columns:1fr 1fr;gap:22px}.subtle{padding:15px 18px;background:#f4f7f9;border-radius:5px}.callout{padding:15px 18px;border-left:3px solid var(--teal);background:#eef6f4;margin:16px 0}.caution{border-left-color:#c08e43;background:#faf5eb}.formula{padding:10px 16px;margin:12px 0;background:#f6f8fa;border:1px solid #e9eef2;border-radius:5px}.formula p{font-size:12px;color:var(--muted);margin:8px 0 2px}.math{overflow-x:auto;overflow-y:hidden;padding:6px 0}.katex-display{margin:.2em 0!important;text-align:left!important}.katex-display>.katex{text-align:left!important}.katex{font-size:1.13em!important}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:5px;margin:14px 0;max-height:560px}table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px}th,td{padding:10px 12px;border-bottom:1px solid #e8edf1;text-align:right}th{background:#eef3f6;position:sticky;top:0;color:#536676;font-weight:600}td:first-child,th:first-child{text-align:left}tr:last-child td{border-bottom:0}tr:hover td{background:#f9fbfc}.tabs{display:flex;gap:7px;flex-wrap:wrap;margin:15px 0}button{font:inherit;font-size:12px;padding:7px 13px;border:1px solid #cbd8df;border-radius:5px;color:var(--ink);background:#fff;cursor:pointer}button.active{background:var(--teal);color:white;border-color:var(--teal)}.tabpanel[hidden]{display:none}.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.metric{border-top:2px solid #afc7cf;padding:10px 0}.metric span{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}.metric strong{font-size:26px;font-weight:600}.step{display:grid;grid-template-columns:36px 1fr;gap:10px;border-top:1px solid var(--line);padding:17px 0}.step .num{color:#8ba7b1;font-size:20px}.step h3{margin:0 0 4px}.step p{margin:4px 0}.step code,code{font:12px/1.5 monospace;color:#385262;background:#eef3f6;padding:2px 4px;border-radius:3px}details{border-top:1px solid var(--line);padding:13px 0;margin-top:10px}summary{cursor:pointer;font-size:13px;font-weight:600;color:#40566b}details pre{max-height:400px;overflow:auto;font-size:11px;white-space:pre-wrap}li{margin:6px 0}.plotly-graph-div{width:100%}footer{font-size:12px;color:var(--muted);padding:10px 0}
@media(min-width:1550px){main{margin-left:calc(210px + (100vw - 1550px)/2)}}
@media(max-width:1050px){aside{width:165px;padding:25px 16px}aside h2{font-size:21px}main{margin-left:165px;padding:24px}.row{grid-template-columns:1fr}}
@media(max-width:700px){aside{position:static;width:auto;padding:14px 18px}aside h2,aside small{display:none}aside .brand{margin:0}aside nav{display:flex;gap:15px;overflow-x:auto}aside a{white-space:nowrap;padding:7px 0}main{margin:0;padding:16px}h1{font-size:29px}.section{padding:18px}.metric-row{grid-template-columns:1fr 1fr}.katex{font-size:1em!important}.row{grid-template-columns:1fr}}
'''

JS='''
document.querySelectorAll('.math').forEach(el=>katex.render(el.dataset.tex,el,{displayMode:true,throwOnError:true,strict:'ignore'}));
document.querySelectorAll('[data-tabs]').forEach(group=>{
 const buttons=group.querySelectorAll(':scope > .tabs > button');
 buttons.forEach(button=>button.addEventListener('click',()=>{
  buttons.forEach(b=>b.classList.toggle('active',b===button));
  group.querySelectorAll(':scope > .tabpanel').forEach(panel=>panel.hidden=panel.id!==button.dataset.target);
  const panel=document.getElementById(button.dataset.target);
  panel.querySelectorAll('.plotly-graph-div').forEach(el=>Plotly.Plots.resize(el));
 }));
});
'''

def build(out,books,trades,stages,search,selected,replays,sensitivity,manifest,ext):
    assets=Path(__file__).parent/'assets'
    h=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>VWAP research — from data to trades</title><style>'+CSS+(assets/'katex.css').read_text()+'</style><script>'+get_plotlyjs()+'</script><script>'+(assets/'katex.js').read_text()+'</script></head><body>']
    h+=['<aside><div class="brand">VWAP WORKSHOP</div><h2>From data<br>to trades.</h2><nav><a href="#data">01 &nbsp; The data</a><a href="#models">02 &nbsp; Model definitions</a><a href="#research">03 &nbsp; Research decisions</a><a href="#trading">04 &nbsp; Trading rule</a><a href="#results">05 &nbsp; Both replays</a><a href="#audit">06 &nbsp; Checks &amp; limits</a></nav><small>Set 1 selects the strategy.<br>Set 2 tests the frozen strategy.<br>One accounting convention.</small></aside><main><header class="masthead"><div class="eyebrow">REBUILT RESEARCH NOTE</div><h1>A book model is only useful<br>if the trade can be executed.</h1><p class="intro">Inspect the observations, test the model changes, freeze one strategy, then account for every fill on both datasets.</p></header>']
    # Data is deliberately the first substantive section; no results precede it.
    h+=['<section class="section" id="data"><div class="kicker">01 / OBSERVATIONS</div><h2>Start with the supplied market data</h2><p>Each replay contains 20 snapshots, five seconds apart, with three price levels on each side. The trade file records aggressor, price, quantity and the snapshot before which each print became known.</p><div data-tabs><div class="tabs"><button class="active" data-target="data1">Set 1 · training</button><button data-target="data2">Set 2 · test</button></div>']
    for i,(label,b) in enumerate(books.items(),1):
        tr=trades[label]
        h += [f'<div class="tabpanel" id="data{i}"'+(' hidden' if i==2 else '')+f'><p class="caption">{b.time.iloc[0]}–{b.time.iloc[-1]} · {len(b)} snapshots · {len(tr)} prints · {int(tr.qty.sum())} traded units</p>',book_timeline(b),'<p class="caption">Bubble area reflects displayed size, with a minimum visible size. Hover for quantity and age. Prices share the same axis. Click legend entries to show or hide series.</p>',details('Inspect one snapshot as a price ladder',ladder(b)),details('View all book fields',table(b.drop(columns=['mid','spread','flow','trade_qty']))),details('View every trade print',table(tr.drop(columns=['signed_qty']))),'</div>']
    h+=['</div><div class="callout"><b>The decision clock.</b> At Tt, use the current book and prints labelled <code>printed_before=Tt</code>. Compute the theo and, if justified, trade immediately against that book. T20 is reserved for flattening. Cash, inventory and flow reset between sets.</div>',math(r'm_t=\frac{b_{1,t}+a_{1,t}}{2},\qquad s_t=a_{1,t}-b_{1,t},\qquad y_t=m_{t+1}','The five-second prediction target exists for T01–T19 only: 19 labelled transitions per set.'),'</section>']
    h+=['<section class="section" id="models"><div class="kicker">02 / DEFINITIONS</div><h2>Keep the mathematics explicit</h2><p>At level i, b and a denote bid and ask prices; q is displayed quantity. These are order-book theoretical prices. They are not transaction VWAPs.</p><h3>VWAP 1 · direct crossed depth</h3>',math(r'V_t^{(1)}=\frac{\sum_{i=1}^3\left(q^a_{i,t}b_{i,t}+q^b_{i,t}a_{i,t}\right)}{\sum_{i=1}^3\left(q^a_{i,t}+q^b_{i,t}\right)}','Opposite-side quantities weight each level’s prices.'),'<h3>VWAP 2 · side averages, then crossed combination</h3>',math(r'\ell=(1,\lambda,\lambda^2),\qquad B_i=\ell_iq_i^b,\quad A_i=\ell_iq_i^a,\quad D_b=\sum_iB_i,\quad D_a=\sum_iA_i'),math(r'\bar b=\frac{\sum_iB_ib_i}{D_b},\quad\bar a=\frac{\sum_iA_ia_i}{D_a},\qquad V^{(2)}=\frac{D_a\bar b+D_b\bar a}{D_a+D_b}','The reference uses λ=0.25. The combination is crossed: more bid depth pushes the theo toward the ask-side average.'),'<h3>VWAP 3 · independent weights and quote age</h3>',math(r'\ell=(1,u,v),\quad 0<v\leq u\leq1;\qquad B_{i,t}=\ell_iq^b_{i,t}d(age^b_{i,t}),\quad A_{i,t}=\ell_iq^a_{i,t}d(age^a_{i,t})'),math(r'd(a)=1,\qquad e^{-a/\tau},\qquad e^{-(a/\tau)^2},\qquad \frac{1}{1+a/\tau}','Use the same side-average and crossed-combination formula with these effective quantities. The fixed reference uses u=0.25, v=0.0625, exponential τ=30s. No decay is an eligible candidate.'),'<h3>Completed-interval trade flow</h3>',math(r'F_t=\frac{\sum_{j\in\mathcal{I}_t}\epsilon_jq_j}{\sum_{j\in\mathcal{I}_t}q_j},\qquad \widehat m_{t+1}=V_t+\gamma F_ts_t','The set Iₜ contains only prints marked available before Tt. BUY has ε=+1 and SELL ε=−1. Missing intervals and T01 have F=0.'),'<p class="caption">Age is used separately for each side and level. Equal decay factors across the entire book cancel. Since deeper prices enter these averages, a theo can lie outside the best bid/ask.</p></section>']
    h+=['<section class="section" id="research"><div class="kicker">03 / TRAINING ONLY</div><h2>Improve the strategy against one objective</h2><p><b>The selection objective is net trading P&amp;L on all of set 1.</b> Every candidate uses the exact same execution policy described next. MAE is reported to explain prediction behaviour, but does not select the final strategy.</p>',math(r'\theta^*=\arg\max_{\theta\in\Theta}\Pi_{\mathrm{net}}^{\mathrm{set\ 1}}(\theta),\qquad \mathrm{MAE}=\frac1{19}\sum_{t=1}^{19}|\widehat m_{t+1}-m_{t+1}|','P&L ties, rounded to 10⁻⁹ GBP, prefer fewer orders, then the earlier candidate. The unchanged model appears first in each incremental search.'),'<div class="callout caution">This is a training fit to a tiny replay. We are explicitly optimising profit, not claiming cross-validated prediction superiority. No test result enters the selection function.</div>']
    frame=pd.DataFrame([dict(Model=s['name'],Net=s['net_pnl'],Gross=s['gross_pnl'],Fees=s['fees'],Orders=s['orders'],MAE=s['mae']) for s in stages])
    h += [table(frame.rename(columns={'Net':'Net P&L (£)','Gross':'Gross P&L (£)','Fees':'Fees (£)','MAE':'MAE (£/unit)'}),digits=4)]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.12,row_heights=[.65,.35])
    fig.add_bar(x=frame.Model,y=frame.Net,name='Set 1 net P&L',marker_color=['#b6c9cf' if s['name']!=selected['name'] else '#09655e' for s in stages],row=1,col=1)
    fig.add_scatter(x=frame.Model,y=frame.MAE,name='Next-midpoint MAE',mode='lines+markers',line_color='#bb8640',row=2,col=1)
    fig.update_yaxes(title_text='P&L (£)',row=1,col=1);fig.update_yaxes(title_text='MAE (£/unit)',row=2,col=1)
    fig.update_xaxes(tickangle=-20,row=2,col=1)
    h += [plot(fig,480,ytitle=None,bottom=100),'<p class="caption">These measures answer different questions. A smaller average prediction error need not produce profitable executable trades.</p>']
    st={s['name']:s for s in stages}
    steps=[('Reference models',f'VWAP 1 earns {money(st["VWAP 1"]["net_pnl"])}. Fixed VWAP 2 earns {money(st["VWAP 2 fixed"]["net_pnl"])}; adding the original age decay gives {money(st["VWAP 3 fixed"]["net_pnl"])}. Therefore complexity alone is not an improvement.'),
           ('Geometric level weights',f'Test λ in {LEVEL2}. The chosen weights give {money(st["Geometric tuned"]["net_pnl"])}. This is a search result, not an assumed improvement.'),
           ('Independent level weights',f'Allow separate u and v, constrained v≤u. Choose u={st["Independent tuned"]["model"]["w2"]:g}, v={st["Independent tuned"]["model"]["w3"]:g}; training net P&L becomes {money(st["Independent tuned"]["net_pnl"])}.'),
           ('Age decay: test and accept or reject',f'Test exponential, Gaussian and hyperbolic decay over τ={TAUS} seconds, alongside no decay. The winning family is <b>{st["Age tuned"]["model"]["decay"]}</b>; net P&L is {money(st["Age tuned"]["net_pnl"])}. The previous model remains eligible.'),
           ('Completed trade flow',f'Test γ in {GAMMAS}. Choose γ={st["Flow tuned"]["model"].get("gamma",0):g}. Training net P&L becomes <b>{money(st["Flow tuned"]["net_pnl"])}</b>. A boundary coefficient is a reason to seek more data, not to assert a universal effect.')]
    for i,(title,body) in enumerate(steps,1):h += [f'<div class="step"><div class="num">{i:02}</div><div><h3>{title}</h3><p>{body}</p></div></div>']
    h += [details('Inspect every search candidate and its training score',table(search,4)),details('Exact model specification at every stage',table(pd.DataFrame([dict(Stage=s['name'],Parameters=json.dumps(s['model'])) for s in stages])))]
    age=search[search.stage=='Age decay'];fig=go.Figure()
    for family in ['exp','gaussian','hyperbolic']:
        points=[(json.loads(r.params)['tau'],r.net_pnl) for r in age.itertuples() if json.loads(r.params).get('decay')==family]
        fig.add_scatter(x=[p[0] for p in points],y=[p[1] for p in points],mode='lines+markers',name=family)
    fig.add_hline(y=st['Independent tuned']['net_pnl'],line_dash='dot',line_color='#52697a')
    fig.update_xaxes(title_text='τ (seconds)')
    h += [details('Age-decay search: why it was retained or rejected',plot(fig,350,'Set 1 net P&L (£)'))]
    flow=search[search.stage=='Trade flow'];points=sorted((json.loads(r.params)['gamma'],r.net_pnl) for r in flow.itertuples())
    fig=go.Figure(go.Scatter(x=[p[0] for p in points],y=[p[1] for p in points],mode='lines+markers',line_color='#09655e',name='Set 1 net P&L'));fig.update_xaxes(title_text='γ')
    h += [details('Trade-flow coefficient search',plot(fig,350,'Set 1 net P&L (£)')),'</section>']
    m=selected['model']
    h+=['<section class="section" id="trading"><div class="kicker">04 / FROZEN STRATEGY</div><h2>Use one model for every final decision</h2><p>The selected model is <b>'+esc(selected['name'])+'</b>. The exact parameters below are frozen before set 2 is loaded by the script. No fallback model is substituted during either replay.</p>',math(r'\ell=(1,'+str(m.get('w2',0))+','+str(m.get('w3',0))+r'),\qquad d(a)=1,\qquad \gamma='+str(m.get('gamma',0))),math(r'\widehat m_{t+1}=\frac{D_a\bar b+D_b\bar a}{D_a+D_b}'+('-' if m.get('gamma',0)<0 else '+')+str(abs(m.get('gamma',0)))+r'F_ts_t','For this fitted strategy, age decay was rejected. The theo is an independent-depth model plus completed-interval flow.'),'<h3>Turn the theo into an executable order</h3><p>At T01–T19, inspect each direction d and integer quantity Q from 1 to 50. Walk asks for a buy and bids for a sell. Reject quantities that exceed displayed liquidity or the position/notional limits. Select the largest positive total surplus; exact ties retain the smaller size.</p>',math(r'x_i=\min\!\left(q_i,\max\!\left(0,Q-\sum_{j<i}x_j\right)\right),\qquad P^{exec}(Q)=\frac{\sum_i x_ip_i^{exec}}{Q}'),math(r'G_t(d,Q)=\sum_i x_i d\left(\widehat m_{t+1}-p_i^{exec}\right)-f-\beta s_tQ','d=+1 for BUY, −1 for SELL. Use f=£0.20 per order and β=0.12. Trade only if max G is strictly positive.'),math(r'Q\leq50,\qquad |I_{t-1}+dQ|\leq100,\qquad |I_{t-1}+dQ|m_t\leq £5{,}000'),'<p>There is at most one order per snapshot. An order may have several level fills. T20 ignores the signal and liquidates the entire position, so its size may exceed 50. If depth cannot flatten it, the script will not report a fully realised terminal P&amp;L.</p><h3>Account for cash, inventory and fees</h3>',math(r'C_t=C_{t-1}-d\sum_i x_ip_i^{exec}-f,\qquad I_t=I_{t-1}+dQ'),math(r'E_t=C_t+I_tm_t,\qquad \Pi_{net}=C_T\ \text{when }I_T=0,\qquad \Pi_{gross}=\Pi_{net}+\sum f'),'<p>Bid/ask crossing and depth impact are already in execution prices; they are not subtracted a second time. Cash uses decimal arithmetic. The fee is per submitted order, not per level. Equity uses midpoint marks; liquidation value uses the remaining executable opposite-side book and an exit fee.</p><div class="callout caution">Execution assumptions: immediate fills, no latency in the base case, and future snapshots unaffected by our trades. The £5,000 limit is a marked-notional proxy because no actual margin schedule was supplied. One unit earns £1 per £1 price change. The entry hurdle reserves the current fee, not an unknown future exit cost.</div></section>']
    h+=['<section class="section" id="results"><div class="kicker">05 / REPLAY</div><h2>Apply that exact strategy to both sets</h2><p>Both sessions start flat. The model, size search, spread buffer, fees, position limits and scheduled liquidation are identical.</p>']
    summary=pd.DataFrame([dict(Dataset=label,**rep[2],**metric(books[label],predict(books[label],selected['model']))) for label,rep in replays.items()])
    h += [table(summary[['Dataset','net_pnl','gross_pnl','fees','orders','terminal_position','mae']].rename(columns={'net_pnl':'Net P&L (£)','gross_pnl':'Gross P&L (£)','fees':'Fees (£)','orders':'Orders','terminal_position':'End inventory','mae':'MAE (£/unit)'}),4)]
    h+=['<div data-tabs><div class="tabs"><button class="active" data-target="replay1">Set 1 · fitted strategy</button><button data-target="replay2">Set 2 · frozen test</button></div>']
    for i,(label,(log,fills,result)) in enumerate(replays.items(),1):
        b=books[label];buys=-fills.loc[fills.side=='BUY','cash_delta'].sum();sales=fills.loc[fills.side=='SELL','cash_delta'].sum()
        h += [f'<div class="tabpanel" id="replay{i}"'+(' hidden' if i==2 else '')+'><div class="metric-row">']
        for title,value in [('Net P&L',money(result['net_pnl'])),('Gross P&L',money(result['gross_pnl'])),('Fees',money(result['fees'])),('Orders',str(result['orders']))]:h += [f'<div class="metric"><span>{title}</span><strong>{value}</strong></div>']
        h += ['</div><h3>Signal, market and actual fills</h3>',result_chart(b,selected,stages,fills),'<p class="caption">The dark teal line is the same selected theo throughout. Arrows mark actual execution prices and sizes. Optional earlier models are hidden by default; toggle them through the legend.</p><h3>Position and P&amp;L through time</h3>',pnl_chart(log),'<h3>Every submitted order</h3>']
        orders=log[log.qty>0][['timestamp','action','qty','avg_price','fee','position','cash','equity']].rename(columns={'timestamp':'Time','action':'Action','qty':'Units','avg_price':'Average price (£)','fee':'Fee (£)','position':'Inventory','cash':'Cash (£)','equity':'Equity (£)'})
        h += [table(orders,3),math(r'\Pi_{net}=\underbrace{'+f'{sales:.2f}'+r'}_{\mathrm{sales}}-\underbrace{'+f'{buys:.2f}'+r'}_{\mathrm{purchases}}-\underbrace{'+f'{result["fees"]:.2f}'+r'}_{\mathrm{fees}}='+f'{result["net_pnl"]:.2f}',f'All values in GBP. End inventory is {result["terminal_position"]}. Peak marked notional {money(result["peak_notional"])}; maximum midpoint-marked drawdown {money(result["max_drawdown"])}.'),details('All 20 decisions, including HOLD and executable surplus',table(log,4)),details('Every level fill: the complete cash ledger',table(fills,4)),'</div>']
    h+=['</div><div class="callout"><b>Read the two results differently.</b> Set 1 profit is the quantity optimised during research. Set 2 measures the unchanged strategy on the designated test replay. Since set 2 has already appeared in previous work, it is a reused holdout, not a new prospective experiment.</div><h3>Execution sensitivity</h3><p>Keep the final model fixed and rerun under each stated execution assumption. These results do not select a replacement model.</p>',table(sensitivity[['dataset','scenario','net_pnl','fees','orders','terminal_position']].rename(columns={'dataset':'Dataset','scenario':'Scenario','net_pnl':'Net P&L (£)','fees':'Fees (£)','orders':'Orders','terminal_position':'End inventory'}),3),'</section>']
    ext_show=ext[['dataset','model','net_pnl','gross_pnl','fees','orders','filled_units','mae']].copy()
    h += ["<section class=\"section\" id=\"extensions\"><div class=\"kicker\">07 / EXTENSIONS</div><h2>Alternative forecasts, thresholds and maximum size</h2><p>These are post-selection diagnostics. The selected strategy remains unchanged. The decision fee changes score G only; actual P&L still charges the stated £0.20 order fee.</p><h3>Last midpoint and blends</h3><p><code>Last midpoint</code> uses the current midpoint as the next forecast. A blend is <code>alpha × selected theo + (1 − alpha) × current midpoint</code>.</p>" + table(ext_show[ext_show.model.str.contains('midpoint|Blend|Selected',case=False,regex=True)].round(5)) + "<h3>Free level weights</h3><p>Replace (1, lambda, lambda²) with independent (1,w1,w2), fixing the first-level weight at 1 and enforcing w2 ≤ w1.</p>" + table(ext_show[ext_show.model.str.startswith('Weights')].round(5)) + "<h3>Trading score and size sensitivity</h3><p>Removing the fee from G changes qualification only. Increasing the clip searches larger quantities, while the position and notional limits remain binding.</p>" + table(ext_show[ext_show.model.str.startswith('Policy') | ext_show.model.str.startswith('Max quantity')].round(5)) + "<div class=\"callout caution\"><b>Maximum quantity.</b> This means the largest fully executable quantity allowed by displayed depth, position and marked notional. The £5,000 limit is a notional proxy, not an exchange margin schedule.</div></section>"]
    h+=['<section class="section" id="audit"><div class="kicker">06 / VERIFICATION</div><h2>What was checked—and what the data cannot establish</h2><div class="row"><div><h3>Numerical checks</h3><ul><li>Input completeness, ordered books, positive integer depth and five-second timestamps.</li><li>Trade availability joins and no use of next-interval prints.</li><li>Original three reference MAEs independently reproduced.</li><li>Every fill bounded by that snapshot’s displayed depth.</li><li>Cash and inventory reconciled at every timestamp.</li><li>Independent FIFO matching agrees with terminal cash P&amp;L.</li><li>Terminal inventory zero; no duplicate spread or fee charge.</li></ul></div><div><h3>Research limits</h3><ul><li>Only 19 transitions per set. Parameter optimisation can overfit.</li><li>Negative flow coefficient is specific to this training fit.</li><li>Quote age does not establish whether liquidity is genuine.</li><li>No queue data, cancellation race, own market impact or borrow costs.</li><li>A one-step theo may create a multi-step holding period.</li><li>No annualised performance or statistical significance claim.</li></ul></div></div><h3>The earlier 50-unit example, corrected</h3>',math(r'P_{sell}=\frac{15(8.55)+20(8.50)+15(8.40)}{50}=8.485,\qquad \Pi=50(8.485-8.50)-0.40=-1.15','This audits the earlier assumed T11-to-T20 trade. It is not the trade path of the newly selected strategy.'),'<h3>Reproduce the work</h3><p>Extract the project, install <code>requirements.txt</code>, then run <code>python research.py --data data --out results</code> and <code>python -m unittest -v test_research.py</code>. The notebook calls these same functions. Model code and presentation code are separated; the report never recalculates its own trading decisions.</p><p>All equations, fonts, plots and controls are embedded. The HTML works offline. Source CSVs are unchanged, and all candidate scores, frozen parameters and fill ledgers are included.</p>',details('Frozen configuration, source hashes and independent reconciliation', '<pre>'+esc(json.dumps(manifest,indent=2))+'</pre>'),'</section><footer>Research order: observations → explicit models → set 1 selection → frozen execution → both replays. Every headline P&amp;L reconciles to the supplied fill ledger.</footer></main><script>'+JS+'</script></body></html>']
    (out/'vwap_research.html').write_text(research_first(''.join(h),books,stages))


def research_first(document, books, stages):
    """Separate predictive research from execution-dependent strategy selection."""
    def section(name):
        start=document.index('<section class="section" id="'+name+'">')
        end=document.index('</section>',start)+len('</section>')
        return document[start:end]
    data=section('data').replace('<b>The decision clock.</b> At Tt, use the current book and prints labelled <code>printed_before=Tt</code>. Compute the theo and, if justified, trade immediately against that book. T20 is reserved for flattening. Cash, inventory and flow reset between sets.',
        '<b>The observation clock.</b> At Tt, the available information is the current book and prints labelled <code>printed_before=Tt</code>. Use it to estimate the midpoint at Tt+1. The next midpoint is not an input. We first study these estimates; the execution rule comes after the VWAP research.')
    b=books['Set 1']; reference=stages[:4]
    scores=pd.DataFrame([dict(Model=stage['name'],**metric(b,predict(b,stage['model']))) for stage in reference])
    prediction=['<section class="section" id="prediction"><div class="kicker">03 / VWAP RESEARCH</div><h2>How well do the VWAPs predict the next midpoint?</h2><p>First evaluate the models as price estimates, using set 1 only. There are no simulated orders in this comparison. At each origin Tt, compare the current model estimate with the observed midpoint at Tt+1.</p>',
        math(r'e_t=V_t-m_{t+1},\qquad \mathrm{MAE}=\frac1{19}\sum_{t=1}^{19}|e_t|,\qquad \mathrm{RMSE}=\sqrt{\frac1{19}\sum_{t=1}^{19}e_t^2}'),
        table(scores.rename(columns={'mae':'MAE (£/unit)','rmse':'RMSE (£/unit)','bias':'Mean error (£/unit)'}),5),
        '<h3>What changes at each model step?</h3><p><b>VWAP 1 → VWAP 2:</b> downweight deeper levels, calculate the two side averages and combine them. Set 1 MAE falls from £0.04654 to £0.04163.</p><p><b>VWAP 2 → fixed VWAP 3:</b> downweight older displayed liquidity. With exponential decay and τ=30s, MAE falls to £0.03726.</p><p><b>Benchmark check:</b> midpoint has MAE £0.02368 and beats all three fixed VWAPs on this measure. These modifications improve the VWAP reference errors, but do not establish superiority over midpoint.</p>']
    fig=go.Figure()
    fig.add_scatter(x=b.timestamp.iloc[:-1],y=b.mid.iloc[1:],name='Next midpoint (target)',line=dict(color='#142d40',width=3))
    for stage in reference:
        fig.add_scatter(x=b.timestamp.iloc[:-1],y=predict(b,stage['model'])[:-1],name=stage['name'],line=dict(color=COLORS[stage['name']]))
    prediction += ['<h3>Forecasts aligned with their targets</h3>',plot(fig,410),'<p class="caption">The horizontal axis is the prediction origin: the target plotted above T01 is the midpoint at T02. This chart evaluates prediction, not execution.</p>']
    fig=go.Figure()
    for stage in reference:
        fig.add_scatter(x=b.timestamp.iloc[:-1],y=predict(b,stage['model'])[:-1]-b.mid.to_numpy()[1:],name=stage['name'],line=dict(color=COLORS[stage['name']]))
    prediction += [plot(fig,350,'Prediction error (£/unit)'),'<h3>Research hypotheses to carry forward</h3><p>Allow independent level weights, compare decay families, and test completed-interval flow. These define the candidate model family. We do not yet choose the final trading model: first we must specify executable prices, size limits and costs.</p><div class="callout"><b>Transition to trading.</b> Average prediction error and executable profit are different objectives. A midpoint forecast cannot justify crossing the spread. We therefore define one trading rule next, then compare model parameters by net set 1 P&amp;L under that rule.</div></section>']
    old_trading=section('trading')
    split=old_trading.index('<h3>Turn the theo into an executable order</h3>')
    frozen_start=old_trading.index('<p>The selected model')
    frozen=old_trading[frozen_start:split]
    execution='<section class="section" id="trading"><div class="kicker">04 / EXECUTION</div><h2>Specify the trading rule before selecting the strategy</h2><p>The following rule accepts any of the candidate theos. Its fees, sizing, spread buffer and risk limits stay fixed for the entire comparison.</p><div class="callout"><b>The trading clock.</b> Observe the Tt book and completed-interval prints, calculate the theo, then execute any justified order against the Tt book. T20 closes the remaining position. Each dataset starts with zero cash P&amp;L, zero inventory and zero initial flow.</div>'+old_trading[split:]
    selection=section('research').replace('id="research"','id="selection"').replace('03 / TRAINING ONLY','05 / STRATEGY SELECTION').replace('execution policy described next','execution policy defined above')
    selection=selection[:-len('</section>')]+'<h3>Freeze the winning strategy</h3>'+frozen+'</section>'
    results=section('results').replace('05 / REPLAY','06 / REPLAY')
    extensions_section=section('extensions').replace('07 / EXTENSIONS','07 / EXTENSIONS')
    audit_section=section('audit').replace('06 / VERIFICATION','08 / VERIFICATION')
    start=document.index('<section class="section" id="data">')
    end=document.index('</section>',document.index('<section class="section" id="audit">'))+len('</section>')
    document=document[:start]+data+section('models')+''.join(prediction)+execution+selection+results+extensions_section+audit_section+document[end:]
    document=document.replace('<a href="#research">03 &nbsp; Research decisions</a>','<a href="#prediction">03 &nbsp; VWAP research</a>')
    document=document.replace('<a href="#results">05 &nbsp; Both replays</a>','<a href="#selection">05 &nbsp; Strategy selection</a><a href="#results">06 &nbsp; Both replays</a>')
    document=document.replace('06 &nbsp; Checks &amp; limits','07 &nbsp; Checks &amp; limits')
    document=document.replace('observations → explicit models → set 1 selection → frozen execution → both replays','observations → VWAP research → execution rule → strategy selection → both replays')
    return document
