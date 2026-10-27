"""Interpretability for the final models (the Overview asks for "clear insight into which technical features or market conditions drive
the signal predictions"). Validation data only (the 8 XFOLDS history blocks, 4,507 cand bars, 1,173 lows); nothing touches test labels.
  1. MC p0: deciles of p0 on cand bars vs the EWMA volatility forecast in units of the 1% barrier, and the actual low rate (calibration).
  2. History GBM: out-of-sample permutation importance per XF block (drop in block AP when one feature is shuffled within the block),
     mean +/- sd over the 8 blocks. The model is refitted at each block start exactly as in solution.py.
Writes explain_mc_deciles.csv and explain_gbm_importance.csv. ~1-2 min."""
import numpy as np, pandas as pd
from common import D, TH, ntr, history, feats, resolve_time, AP
from solution import gbm_fit_predict, prep
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

X = history(); xp, OFF, XF, yx = X['xp'], X['OFF'], X['XFOLDS'], X['yx']; cand = X['Z']['cand'].astype(bool); n = len(xp); N = OFF + ntr
mc = pd.read_csv(D + 'oofx/xmc_B3.csv').score.values
m = np.zeros(n, bool)
for a, b in XF: m[a:b] = True
m &= cand
sig = prep(xp, 12)[0]                                                   # EWMA (HL 12) 1-hour vol forecast, known at t
d = pd.DataFrame({'p0': mc[m], 'sig_over_TH': sig[m] / TH, 'low': yx[m]})
d['decile'] = pd.qcut(d.p0.rank(method='first'), 10, labels=False) + 1
T1 = d.groupby('decile').agg(bars=('low', 'size'), mean_p0=('p0', 'mean'), actual_low_rate=('low', 'mean'), mean_vol_over_1pct=('sig_over_TH', 'mean')).round(3)
T1.to_csv(D + 'explain_mc_deciles.csv', encoding='utf-8')
print(f'MC p0 on {m.sum()} history cand bars: Spearman(p0, vol/1%) = {d.p0.corr(d.sig_over_TH, method="spearman"):.3f}')
print(T1.to_string())

F = feats(xp, X['xhour'], X['xdow']); names = list(F.columns[:17]); F17 = F.iloc[:, :17].values
s19 = int(X['xdt'].searchsorted(pd.Timestamp('2019-01-01')))
cidx = np.flatnonzero(cand[s19:N]) + s19; tau = np.full(n, -1); tau[cidx] = resolve_time(xp[:N], cidx)
rng = np.random.default_rng(0); imp = np.zeros((len(XF), len(names)))
for k, (a, b) in enumerate(XF):
    tri = np.arange(s19, a); tri = tri[tau[tri] < a]
    mdl = HGB(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=0).fit(F17[tri], yx[tri])
    Xb = F17[a:b].copy(); base = AP(yx[a:b], mdl.predict_proba(Xb)[:, 1] * cand[a:b])
    assert np.allclose(mdl.predict_proba(Xb)[:, 1] * cand[a:b], gbm_fit_predict(F17, yx, cand, tau, s19, a, b))   # same model as solution.py
    for j in range(len(names)):
        drops = []
        for _ in range(3):
            Xp = Xb.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); drops.append(base - AP(yx[a:b], mdl.predict_proba(Xp)[:, 1] * cand[a:b]))
        imp[k, j] = np.mean(drops)
T2 = pd.DataFrame({'feature': names, 'importance_mean': imp.mean(0), 'importance_sd_over_blocks': imp.std(0), 'blocks_positive': (imp > 0).sum(0)})
T2 = T2.sort_values('importance_mean', ascending=False).reset_index(drop=True)
T2.to_csv(D + 'explain_gbm_importance.csv', index=False, encoding='utf-8')
print('\nHistory GBM: permutation importance (drop in block AP), 8 XF blocks x 3 shuffles'); print(T2.round(4).to_string(index=False))
