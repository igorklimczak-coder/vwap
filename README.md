# VWAP research rebuilt

Open results/vwap_research.html. It contains the data first, model definitions, all incremental searches, the frozen execution rule, and both replays. KaTeX and Plotly work offline.

Install requirements.txt, then run:

    python research.py --data data --out results
    python -m unittest -v test_research.py

The selected model maximises full set 1 net P&L, with ties preferring fewer orders then earlier candidates. Weights (1, 1, 0.0625), no age decay, flow coefficient -0.5. Execution constants remain fixed.

Set 1: £7.85 gross minus £1.60 fees = £6.25 net.
Set 2: £4.00 gross minus £1.00 fees = £3.00 net.
Both finish flat. Set 1 is an optimised training result. Set 2 is a reused holdout, not a fresh prospective trial.

research.py owns all calculations; report_builder.py owns presentation. Keep both and assets/ together. The notebook calls the same numerical functions; its cells were executed sequentially in IPython. Eight tests cover reference formulas, flow timing, depth, cash/FIFO accounting, causality and incomplete liquidation.

Immediate displayed-depth execution is assumed; our fills do not alter future recorded snapshots. £0.20 per order, at most 50 units per ordinary order, position 100, marked notional £5,000, spread buffer 0.12. T20 liquidates the full position. Full assumptions and sensitivities are in the report.

Unconstrained weights: see results/unconstrained_weights.csv and the final report.


## Data split discipline
Set 1 is the only calibration/training set. VWAP definitions, weight grids, decay, flow coefficient, trading thresholds, and quantity rules are selected using Set 1. Set 2 is loaded only after the frozen configuration is written and is used once for out-of-sample evaluation. The unconstrained-weight Set-2 table is a diagnostic, never a selection criterion.
