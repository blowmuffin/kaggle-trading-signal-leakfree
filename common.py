"""Shared data, folds and helpers. New scripts import this instead of copying code (the older scripts predate it and stay self-contained).
Import is cheap: it only loads the CSVs. Everything at bar i must use prices[0..i] only."""
import os
from functools import cache
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP

D = os.path.dirname(os.path.abspath(__file__)) + os.sep
OOF = D + 'oof' + os.sep
tr = pd.read_csv(D + 'btc_train.csv', encoding='utf-8')
te = pd.read_csv(D + 'btc_test.csv', encoding='utf-8')
sub = pd.read_csv(D + 'sample_submission.csv', encoding='utf-8')
assert (tr.Id.values == np.arange(len(tr))).all() and te.Id.iloc[0] == len(tr)
ntr = len(tr)
allp = np.concatenate([tr.prices.values, te.prices.values]).astype(float)
nall = len(allp)
dt = pd.to_datetime(pd.concat([tr.datetime, te.datetime]).values)
hour, dow = dt.hour.values, dt.dayofweek.values
y0 = tr.Signal.values.astype(int)
TH = 0.01
FOLDS = [(int(ntr * (0.5 + 0.1 * k)), int(ntr * (0.6 + 0.1 * k))) for k in range(5)]   # the ONE fold definition; validation = [3066, 6132)


def zz(p, label_until=None):
    """Causal ZigZag(1%), copied from probe_mc.py. Row i = state AFTER bar i: M mode (0 undecided / 1 up / -1 down), E running extreme,
    cand = bar i is the running low of a down-swing. `lows` = confirmed pivot lows (uses future bars: labels on train ONLY, never a feature).
    label_until: record only lows confirmed at a bar < label_until, so the low rule is never evaluated on later (test) prices."""
    n = len(p)
    cand = np.zeros(n); dd = np.zeros(n); hfh = np.zeros(n); nl = np.zeros(n); M = np.zeros(n, int); E = np.zeros(n, int)
    lows = []; mode, ext, hi, nlows = 0, 0, 0, 0
    for i in range(n):
        if i >= 1:
            if mode == 0:
                if p[i] >= p[ext] * (1 + TH): lows += [ext] if label_until is None or i < label_until else []; mode = 1; ext = i
                elif p[i] <= p[ext] * (1 - TH): mode = -1; hi = 0; ext = i; nlows = 0
                elif p[i] < p[ext]: ext = i
            elif mode == 1:
                if p[i] > p[ext]: ext = i
                elif p[i] <= p[ext] * (1 - TH): mode = -1; hi = ext; ext = i; nlows = 0
            else:
                if p[i] < p[ext]: ext = i; nlows += 1
                elif p[i] >= p[ext] * (1 + TH): lows += [ext] if label_until is None or i < label_until else []; mode = 1; ext = i
        cand[i] = mode == -1 and ext == i
        dd[i] = np.log(p[i] / p[hi]); hfh[i] = i - hi; nl[i] = nlows; M[i] = mode; E[i] = ext
    return dict(cand=cand, dd=dd, hfh=hfh, nl=nl, M=M, E=E, lows=lows)


def fold_ap(s, y=y0):
    """Per-fold AP of a full-length (nall) score series."""
    return np.array([AP(y[a:b], s[a:b]) for a, b in FOLDS])


def sub_check(out):
    assert (out.Id.values == sub.Id.values).all() and len(out) == 2628
    assert not out.isna().any().any() and ((out.Signal >= 0) & (out.Signal <= 1)).all()


def save_oof(name, s):
    """Store a model for team.py. s = full-length (nall) series: rows [3066, ntr) out-of-fold on FOLDS, test rows = the final-fit scores."""
    assert len(s) == nall and not np.isnan(s[FOLDS[0][0]:]).any()
    os.makedirs(OOF, exist_ok=True)
    pd.DataFrame({'Id': np.arange(nall), 'score': s}).to_csv(OOF + name + '.csv', index=False, encoding='utf-8')


# ---------------------------------------------------------------- Binance history (external/, plan Phase 1). Lazy: import stays cheap.
@cache
def history():
    """ONE contiguous series = Binance history (2017-08 -> train start) + the competition series: xp[OFF + i] == allp[i].
    yx = ZigZag(1%) lows on the continuous series, recorded only when confirmed before the test start (label_until), so test-period
    labels are never computed; on train it equals y0 except row 4, a low only because the official ZigZag restarts at train row 0.
    Z = the causal ZigZag state over train + test. ohlcv rows align with xp[:OFF + ntr] (no test-period OHLCV on disk).
    XFOLDS = 8 half-year history blocks 2022H1 .. 2025-07 -> train start (validation only)."""
    k = pd.read_csv(D + 'external/btcusdt_1h.csv', encoding='utf-8', parse_dates=['datetime'])
    OFF = int((k.datetime < dt[0]).sum())
    assert len(k) == OFF + ntr and k.datetime[OFF - 1] == dt[0] - pd.Timedelta('1h') and (k.close.values[OFF:] == allp[:ntr]).all()
    xp = np.r_[k.close.values[:OFF], allp]
    xdt = pd.DatetimeIndex(k.datetime[:OFF]).append(dt)
    Zx = zz(xp, label_until=OFF + ntr); yx = np.zeros(len(xp), int); yx[Zx['lows']] = 1
    edges = list(pd.date_range('2022-01-01', '2025-07-01', freq='6MS')) + [dt[0]]
    XFOLDS = [(int(xdt.searchsorted(a)), int(xdt.searchsorted(b))) for a, b in zip(edges[:-1], edges[1:])]
    assert len(XFOLDS) == 8 and XFOLDS[-1][1] == OFF
    return dict(OFF=OFF, xp=xp, xdt=xdt, xhour=xdt.hour.values, xdow=xdt.dayofweek.values, yx=yx, Z=Zx, XFOLDS=XFOLDS, ohlcv=k)


def xfold_ap(s):
    """Per-block AP on XFOLDS of a full-length (len(xp)) score series."""
    X = history(); return np.array([AP(X['yx'][a:b], s[a:b]) for a, b in X['XFOLDS']])


def xf_bootstrap(s, ref, reps=400, seed=0):
    """Paired bootstrap of mean-XF-AP(s) - mean-XF-AP(ref): resample calendar days within each block. Returns the reps differences."""
    X = history(); yx = X['yx']; day = np.asarray((X['xdt'] - X['xdt'][0]).days); groups = []
    for a, b in X['XFOLDS']:
        cut = np.r_[0, np.flatnonzero(np.diff(day[a:b])) + 1, b - a]; groups.append([np.arange(a + cut[i], a + cut[i + 1]) for i in range(len(cut) - 1)])
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        rows = [np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for g in groups]
        out.append(np.mean([AP(yx[r], s[r]) - AP(yx[r], ref[r]) for r in rows]))
    return np.array(out)


def save_oofx(name, s):
    """Store a model's history scores for team.py: s = len(xp) series, scored on the XFOLDS span (out-of-block for fitted models)."""
    X = history(); a0, a1 = X['XFOLDS'][0][0], X['OFF']
    assert len(s) == len(X['xp']) and not np.isnan(s[a0:a1]).any()
    os.makedirs(D + 'oofx', exist_ok=True)
    pd.DataFrame({'i': np.arange(len(s)), 'score': s}).to_csv(D + 'oofx/' + name + '.csv', index=False, encoding='utf-8')


def resolve_time(p, idx):
    """Bar at which each cand bar's fate is known: first later close < p_s (no low) or >= 1.01 p_s (confirmed); len(p) if unresolved
    within p. Pass p truncated at the last bar you may look at; a training row is usable for a fit starting at a only if this is < a."""
    out = np.full(len(idx), len(p))
    for j, s in enumerate(idx):
        seg = p[s + 1:s + 3001]; hit = (seg < p[s]) | (seg >= p[s] * (1 + TH))
        if hit.any(): out[j] = s + 1 + hit.argmax()
    return out


def ewma_sig(p, HL):
    """sig[t] = EWMA scale forecast for r[t+1] (as probe_mc2.prep_base). Causal."""
    r = np.r_[np.nan, np.diff(np.log(p))]; lam = 0.5 ** (1.0 / HL); s2 = np.full(len(p), np.nan)
    if len(p) > 50:
        v = np.mean(r[1:51] ** 2); s2[50] = v
        for t in range(51, len(p)): v = lam * v + (1 - lam) * r[t] ** 2; s2[t] = v
    return np.sqrt(s2)


def feats(p, hr, dw):
    """make_sub.py's 17 BASE features + lsig12 (log EWMA sigma, HL 12). Row i uses p[:i+1] only."""
    Zs = zz(p, label_until=0); lp = pd.Series(np.log(p)); r1 = lp.diff()
    F = pd.DataFrame({'cand': Zs['cand'], 'dd': Zs['dd'], 'hrs_from_hi': Zs['hfh'], 'nlows': Zs['nl']})
    for k in (1, 2, 3, 6, 12, 24): F[f'ret{k}'] = lp.diff(k)
    for k in (6, 24, 72): F[f'vol{k}'] = r1.rolling(k).std()
    d = pd.Series(p).diff(); g = d.clip(lower=0).rolling(14).mean(); l = (-d).clip(lower=0).rolling(14).mean()
    F['rsi'] = g / (g + l); F['dist_sma24'] = lp - lp.rolling(24).mean(); F['hour'] = hr; F['dow'] = dw
    F['lsig12'] = np.log(ewma_sig(p, 12))
    return F.replace([np.inf, -np.inf], np.nan).fillna(0)
