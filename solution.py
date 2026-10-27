"""Final pipeline. Rebuilds both final submissions from the raw data, with no-look-ahead checks, and verifies them against the submitted files.

  pick 1  final/pick1_blend_coins.csv == submission_blend_mc_coins.csv (Kaggle sub 57156500)
          0.5 * MC p0 + 0.5 * a gradient-boosted model trained on 2019+ BTC history PLUS ETH/BNB/SOL/XRP/ADA history.
  pick 2  final/pick2_blend_btc.csv   == submission_blend_mc_gbm.csv   (Kaggle sub 57007260)
          0.5 * MC p0 + 0.5 * the same gradient-boosted model trained on 2019+ BTC history only.
  pick 2b final/pick2b_blend_btc_intrabar.csv == submission_blend_mc_gbm_ib.csv (10-22)
          0.5 * MC p0 + 0.5 * the BTC-only model with 5 intrabar features of each bar's own minutes (fetch_m1.py: last 5 / 15 minutes'
          return, minutes since the hour's low, last 5 minutes' taker-buy share and volume share), all closed by the bar's close.
  pick 3  final/pick3_avg_1_2b.csv == submission_avg_p1_p2b.csv (10-24): 0.5 * pick 1 + 0.5 * pick 2b, the best single file
          (XF 0.3368, FOLDS 0.3534; gate v2 vs pick 1), but the PAIR pick 1 + pick 2b has the higher E[max], so it is not a final pick.
  pick 4  final/pick4_p1star.csv == submission_p1star.csv (10-25, probe_r7.py): 0.4 * MCL + 0.6 * a gradient-boosted model trained on
          BTC + ETH/BNB/SOL/XRP/ADA history in which EVERY row carries its own asset's 5 intrabar features (the coins' from fetch_m1.py
          SYMBOL, history only). MCL = the MC with 36 more runs on trailing pools W 960 / 1920 (90 runs). Confirmed on 10 coins never used
          to choose these components (+0.0051 AP over pick 1 on 80 coin-blocks, 55/80; the gain is mostly the coins' own intrabar rows).
          BTC XF +0.0077 (the data the components were screened on, so not a confirmation); pair with pick 2b +0.003 E[max].
  MC p0 = probability that a candidate bar is confirmed as a ZigZag(1%) low, 54 simulation runs on the Binance-history + competition series.

Inputs: btc_train.csv, btc_test.csv, sample_submission.csv (competition) and external/{btc,eth,bnb,sol,xrp,ada}usdt_1h.csv
(python fetch_external.py [SYMBOL]; Binance 1h klines from data.binance.vision, all stopping before the test start), and for pick 2b
external/m1feat_btcusdt.csv + external/m1feat_btcusdt_test.csv (python fetch_m1.py; python fetch_m1.py --test: hourly summaries of
Binance BTCUSDT 1m klines, the test-period file kept separate), and for pick 4 also external/m1feat_{eth,bnb,sol,xrp,ada}usdt.csv
(python fetch_m1.py ETHUSDT etc.; history only). Shared code: common.py.
  python solution.py         both submissions + leak tests + reproduction check   (~1 min on 12 cores)
  python solution.py --cv    also the validation tables of SOLUTION.md            (~3 min on 12 cores)
  python solution.py --audit also an end-to-end look-ahead audit: both picks re-scored at 6 test bars from the series cut at that bar
                             (MC, features, ZigZag state and GBM all rebuilt from prices[:t+1]) must equal the submitted values (~1 min)
No look-ahead: every score at bar t is a function of prices[0..t] only; training labels (BTC and coins) are used only once resolved before
the fit start; the ZigZag-low rule is never evaluated on test prices (zz(label_until=...)). Test predictions of picks 1 and 2 use
competition BTC prices only; pick 2b also uses each test bar's own 1m klines (all closed by that bar's close; fetch_m1.py --test)."""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from common import (D, TH, allp, ntr, nall, te, FOLDS, zz, fold_ap, xfold_ap, xf_bootstrap, history, feats, resolve_time, sub_check)

GRID = [(W, HL) for W in (120, 240, 480) for HL in (6, 12, 24)]     # trailing pool window W (bars) x EWMA half-life HL (bars)
GRIDL = [(W, HL) for W in (960, 1920) for HL in (6, 12, 24)]         # pick 4's MCL adds these longer pools
NSIM, H, L = 1000, 72, 3                                             # paths per bar, horizon (bars), bootstrap block length
COINS = ('ethusdt', 'bnbusdt', 'solusdt', 'xrpusdt', 'adausdt')      # extra training rows for the pick-1 GBM


# ------------------------------------------------------------------ MC p0 (filtered historical simulation)
def prep(p, HL):
    """sig[t] = EWMA (half-life HL) scale forecast for r[t+1]; zp[s] = r_s / sig[s-1] = returns standardised by the scale known before
    them. Both causal (seeded with the first 50 returns)."""
    r = np.r_[np.nan, np.diff(np.log(p))]; lam = 0.5 ** (1.0 / HL); s2 = np.full(len(p), np.nan)
    v = np.mean(r[1:51] ** 2); s2[50] = v
    for t in range(51, len(p)): v = lam * v + (1 - lam) * r[t] ** 2; s2[t] = v
    sig = np.sqrt(s2); zp = np.full(len(p), np.nan); zp[51:] = r[51:] / sig[50:-1]
    return sig, zp


def p0_bar(p, sig, zp, t, W, seed):
    """P(cand bar t is confirmed as a low): NSIM paths of H steps from p[t], each step sig[t] * z with z drawn in blocks of L consecutive
    values from the trailing W standardised returns (demeaned: zero drift). From a cand bar (the running low of a down-swing), bar t is
    confirmed iff a path closes >= p_t * (1 + TH) before any close < p_t. Uses p[:t+1], sig[t], zp[:t+1] only; rng seeded by seed + t."""
    lo = max(51, t - W + 1); z = zp[lo:t + 1]; z = z - z.mean()
    rng = np.random.default_rng(seed + t); nb = -(-H // L)
    idx = (rng.integers(0, len(z) - L + 1, size=(NSIM, nb))[:, :, None] + np.arange(L)).reshape(NSIM, nb * L)[:, :H]
    Q = p[t] * np.exp(np.cumsum(sig[t] * z[idx], axis=1))
    fail, succ = Q < p[t], Q >= p[t] * (1 + TH)
    ff = np.where(fail.any(1), fail.argmax(1), H); fs = np.where(succ.any(1), succ.argmax(1), H + 1)
    return (fs < ff).mean()


def _run(args):
    W, HL, seed, ts = args
    p = history()['xp']; sig, zp = prep(p, HL)
    return np.array([p0_bar(p, sig, zp, t, W, seed) for t in ts])


def mc(ts, ex, grid=GRID):
    """Average p0 over grid x seeds 0-5 at bars ts of the history + competition series."""
    return np.mean(list(ex.map(_run, [(W, HL, sd, ts) for sd in range(6) for W, HL in grid])), axis=0)


# ------------------------------------------------------------------ history GBM (pick 2: BTC rows; pick 1: + coin rows)
def coin_rows(cut, coins=COINS):
    """17 features, ZigZag labels and the time each label becomes known, for every coin bar from 2019 (files stop before the test start).
    Non-cand bars are never lows (label known at once); a cand bar's label is known at the bar that closes >= 1.01 p_s or < p_s."""
    CF, CY, CD, CK = [], [], [], []
    for c in coins:
        k = pd.read_csv(D + f'external/{c}_1h.csv', parse_dates=['datetime']); k = k[k.datetime >= '2019-01-01'].reset_index(drop=True)
        assert k.datetime.max() < cut, f'{c}: test-period row'
        p, dt = k.close.values, pd.DatetimeIndex(k.datetime)
        Zc = zz(p); y = np.zeros(len(p), int); y[Zc['lows']] = 1
        known = dt.values.copy(); ci = np.flatnonzero(Zc['cand']); r = resolve_time(p, ci)
        known[ci] = np.where(r < len(p), dt.values[np.minimum(r, len(p) - 1)], np.datetime64('2262-01-01'))
        F = feats(p, dt.hour.values, dt.dayofweek.values).iloc[:, :17].values
        i = len(p) // 2; assert np.allclose(feats(p[:i + 1], dt.hour.values[:i + 1], dt.dayofweek.values[:i + 1]).iloc[i, :17].values, F[i]), f'LOOK-AHEAD {c}'
        CF.append(F); CY.append(y); CD.append(dt.values); CK.append(known)
    return np.vstack(CF), np.concatenate(CY), np.concatenate(CD), np.concatenate(CK)


def gbm_fit_predict(F17, yx, cand, tau, s19, a, b, coins=None, Ta=None, coin_weight=None, hgb=None, est=None):
    """Fit on BTC rows [s19, a) whose labels are resolved before a (cand rows: tau < a; non-cand rows: label 0 known at once), plus, if
    given, coin rows timestamped and resolved before Ta (the time of bar a), optionally with sample weight coin_weight (BTC rows: 1);
    predict BTC rows [a, b), times cand. hgb: optional HGB settings overriding the defaults; est: optional estimator factory replacing HGB (research only)."""
    tri = np.arange(s19, a); tri = tri[tau[tri] < a]
    assert tri.max() < a and tau[tri].max() < a
    Xtr, ytr, w = F17[tri], yx[tri], None
    if coins is not None:
        CF, CY, CD, CK = coins; ok = (CD < Ta) & (CK < Ta); assert CK[ok].max() < Ta
        Xtr, ytr = np.r_[Xtr, CF[ok]], np.r_[ytr, CY[ok]]
        if coin_weight is not None: w = np.r_[np.ones(len(tri)), np.full(ok.sum(), coin_weight)]
    m = est() if est else HGB(**{**dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=0), **(hgb or {})})
    return m.fit(Xtr, ytr, sample_weight=w).predict_proba(F17[a:b])[:, 1] * cand[a:b]


def coin_intrabar(cut, coins=COINS):
    """The 5 intrabar features for every coin_rows() row (same order: each coin's 2019+ bars), from external/m1feat_<coin>.csv (history
    only; each row summarises that bar's own minutes); NaN where a bar has none."""
    out = []
    for c in coins:
        dt = pd.DatetimeIndex(pd.read_csv(D + f'external/{c}_1h.csv', parse_dates=['datetime']).datetime)
        f = pd.read_csv(D + f'external/m1feat_{c}.csv', parse_dates=['datetime']).set_index('datetime')
        assert f.index.max() < cut, f'{c}: test-period minutes'
        out.append(f.reindex(dt[dt >= '2019-01-01'])[['m_ret5', 'm_ret15', 'm_lowage', 'm_tbs5', 'm_vsh5']].values)
    return np.vstack(out)


def intrabar(xdt, N):
    """The 5 intrabar features per BTC bar (rows of xp): history file for bars before the test start, test file for the test bars.
    Each row summarises only that bar's own minutes (fetch_m1.py checks the last minute's close == the official close); NaN if absent."""
    h = pd.read_csv(D + 'external/m1feat_btcusdt.csv', parse_dates=['datetime']).set_index('datetime')
    t = pd.read_csv(D + 'external/m1feat_btcusdt_test.csv', parse_dates=['datetime']).set_index('datetime')
    assert h.index.max() < xdt[N] and (t.index == pd.DatetimeIndex(xdt[N:])).all(), 'intrabar files misaligned with the series'
    c = ['m_ret5', 'm_ret15', 'm_lowage', 'm_tbs5', 'm_vsh5']                   # pick 2b's features (the files may hold more)
    return np.r_[h.reindex(xdt[:N])[c].values, t[c].values]


if __name__ == '__main__':
    T_START = time.time(); CV = '--cv' in sys.argv; AUDIT = '--audit' in sys.argv
    os.makedirs(D + 'final', exist_ok=True)
    cand_c = zz(allp, label_until=ntr)['cand'].astype(bool)                   # causal ZigZag state; no low is recorded on test prices
    X = history(); xp, OFF, XF, yx, xdt = X['xp'], X['OFF'], X['XFOLDS'], X['yx'], X['xdt']
    cand_x = X['Z']['cand'].astype(bool); n = len(xp); N = OFF + ntr
    assert (cand_x[OFF:] == cand_c).all() and yx[N:].sum() == 0 and max(X['Z']['lows']) < N, 'state mismatch or a test-period label'

    xs = np.arange(n); inx = np.zeros(n, bool)
    if CV:
        for a, b in XF: inx[a:b] = True
    ts = np.flatnonzero(cand_x & (inx | (xs >= (OFF + FOLDS[0][0] if CV else N))))
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        m2 = np.zeros(n); m2[ts] = mc(ts, ex)
        mL = np.zeros(n); mL[ts] = (len(GRID) * m2[ts] + len(GRIDL) * mc(ts, ex, GRIDL)) / (len(GRID) + len(GRIDL))   # 90-run mean
    print(f'MC done: {len(ts)} cand bars [{time.time() - T_START:.0f}s]', flush=True)

    # ---- leak test: MC from the truncated series at the first and last test cand bars, one setting each
    full_sig, full_zp = prep(xp, 12)
    for t in (int(ts[ts >= N][0]), int(ts[-1])):
        sig, zp = prep(xp[:t + 1], 12)
        assert p0_bar(xp[:t + 1], sig, zp, t, 240, 4) == p0_bar(xp, full_sig, full_zp, t, 240, 4), f'LOOK-AHEAD MC {t}'
        assert p0_bar(xp[:t + 1], sig, zp, t, 1920, 4) == p0_bar(xp, full_sig, full_zp, t, 1920, 4), f'LOOK-AHEAD MCL {t}'
    # ---- history GBMs: 17 causal close features, 2019+ rows, purged labels
    F17 = feats(xp, X['xhour'], X['xdow']).iloc[:, :17].values
    for i in (OFF - 1, N, n - 1):
        assert np.allclose(feats(xp[:i + 1], X['xhour'][:i + 1], X['xdow'][:i + 1]).iloc[i, :17].values, F17[i]), f'LOOK-AHEAD feats {i}'
    s19 = int(xdt.searchsorted(pd.Timestamp('2019-01-01')))
    cidx = np.flatnonzero(cand_x[s19:N]) + s19
    tau = np.full(n, -1); tau[cidx] = resolve_time(xp[:N], cidx)              # resolution decided from pre-test prices only
    assert not yx[s19:N][~cand_x[s19:N]].any()
    coins = coin_rows(xdt[N])
    g = np.full(n, np.nan); gc = np.full(n, np.nan)
    g[N:] = gbm_fit_predict(F17, yx, cand_x, tau, s19, N, n)
    gc[N:] = gbm_fit_predict(F17, yx, cand_x, tau, s19, N, n, coins, xdt[N].to_datetime64())
    FM = intrabar(xdt, N); FI = np.c_[F17, FM]; gi = np.full(n, np.nan)
    assert np.isfinite(FM[N:][cand_x[N:]]).all(), 'a test cand bar has no intrabar features'
    gi[N:] = gbm_fit_predict(FI, yx, cand_x, tau, s19, N, n)
    coins_ib = (np.c_[coins[0], coin_intrabar(xdt[N])],) + coins[1:]                      # pick 4: every row with its own intrabar
    gci = np.full(n, np.nan); gci[N:] = gbm_fit_predict(FI, yx, cand_x, tau, s19, N, n, coins_ib, xdt[N].to_datetime64())
    print(f'LEAK TESTS PASSED (MC truncated series at test bars, BTC + coin features, purge, no test labels, coin files before test start) '
          f'[{time.time() - T_START:.0f}s]', flush=True)

    # ---- submissions + reproduction check
    picks = {'pick1_blend_coins.csv': (0.5 * m2[N:] + 0.5 * gc[N:], 'submission_blend_mc_coins.csv'),
             'pick2_blend_btc.csv': (0.5 * m2[N:] + 0.5 * g[N:], 'submission_blend_mc_gbm.csv'),
             'pick2b_blend_btc_intrabar.csv': (0.5 * m2[N:] + 0.5 * gi[N:], 'submission_blend_mc_gbm_ib.csv')}
    picks['pick3_avg_1_2b.csv'] = (0.5 * picks['pick1_blend_coins.csv'][0] + 0.5 * picks['pick2b_blend_btc_intrabar.csv'][0], 'submission_avg_p1_p2b.csv')
    picks['pick4_p1star.csv'] = (0.4 * mL[N:] + 0.6 * gci[N:], 'submission_p1star.csv')
    for fn, (sig_, ref) in picks.items():
        out = pd.DataFrame({'Id': te.Id.values, 'Signal': sig_}); sub_check(out)
        out.to_csv(D + 'final/' + fn, index=False, encoding='utf-8')
        if os.path.exists(D + ref):
            d = np.abs(out.Signal.values - pd.read_csv(D + ref).Signal.values).max()
            assert d < 1e-12, f'{fn} differs from the submitted {ref} by {d}'
            print(f'final/{fn}: REPRODUCES {ref} (max abs diff {d:.1e})')
        else: print(f'final/{fn} written ({ref} not present to compare)')

    if AUDIT:   # nothing after bar t exists in these recomputations, so any look-ahead would change the score
        tc = np.flatnonzero(cand_x[N:]) + N
        for t in [N] + [int(x) for x in tc[np.linspace(0, len(tc) - 1, 5).astype(int)]]:     # the boundary row + 5 cand bars
            p = xp[:t + 1]; Zt = zz(p, label_until=N); ct = Zt['cand'].astype(bool); mt = 0.0
            yt = np.zeros(t + 1, int); yt[Zt['lows']] = 1                            # labels and purge times rebuilt from the cut series
            ci = np.flatnonzero(ct[s19:N]) + s19; taut = np.full(t + 1, -1); taut[ci] = resolve_time(p[:N], ci)
            mLt = 0.0
            if ct[t]:
                pr = {HL: prep(p, HL) for HL in (6, 12, 24)}
                mt = np.mean([p0_bar(p, *pr[HL], t, W, sd) for sd in range(6) for W, HL in GRID])
                mLt = np.mean([p0_bar(p, *pr[HL], t, W, sd) for sd in range(6) for W, HL in GRID + GRIDL])
            Ft = feats(p, X['xhour'][:t + 1], X['xdow'][:t + 1]).iloc[:, :17].values
            gt = gbm_fit_predict(Ft, yt, ct, taut, s19, N, t + 1)[-1]
            gct = gbm_fit_predict(Ft, yt, ct, taut, s19, N, t + 1, coins, xdt[N].to_datetime64())[-1]
            git = gbm_fit_predict(np.c_[Ft, FM[:t + 1]], yt, ct, taut, s19, N, t + 1)[-1]      # intrabar rows <= t only
            gcit = gbm_fit_predict(np.c_[Ft, FM[:t + 1]], yt, ct, taut, s19, N, t + 1, coins_ib, xdt[N].to_datetime64())[-1]
            for fn, v in (('pick1_blend_coins.csv', 0.5 * mt + 0.5 * gct), ('pick2_blend_btc.csv', 0.5 * mt + 0.5 * gt),
                          ('pick2b_blend_btc_intrabar.csv', 0.5 * mt + 0.5 * git), ('pick3_avg_1_2b.csv', 0.5 * mt + 0.25 * gct + 0.25 * git),
                          ('pick4_p1star.csv', 0.4 * mLt + 0.6 * gcit)):
                assert abs(v - picks[fn][0][t - N]) < 1e-12, f'AUDIT FAIL {fn} at test Id {t - OFF}: {v} vs {picks[fn][0][t - N]}'
            print(f'AUDIT test Id {t - OFF} (cand {int(ct[t])}): pick 1 {0.5 * mt + 0.5 * gct:.6f}, pick 2 {0.5 * mt + 0.5 * gt:.6f}, '
                  f'pick 2b {0.5 * mt + 0.5 * git:.6f}, pick 4 {0.4 * mLt + 0.6 * gcit:.6f} rebuilt from prices[:t+1] only == the full-run output [{time.time() - T_START:.0f}s]', flush=True)
        print('END-TO-END AUDIT PASSED: all picks at 6 test bars equal their recomputation from the truncated series')

    if CV:   # validation tables (SOLUTION.md)
        for a, b in XF + [(OFF + fa, OFF + fb) for fa, fb in FOLDS]:
            g[a:b] = gbm_fit_predict(F17, yx, cand_x, tau, s19, a, b)
            gc[a:b] = gbm_fit_predict(F17, yx, cand_x, tau, s19, a, b, coins, xdt[a].to_datetime64())
            gi[a:b] = gbm_fit_predict(FI, yx, cand_x, tau, s19, a, b)
            gci[a:b] = gbm_fit_predict(FI, yx, cand_x, tau, s19, a, b, coins_ib, xdt[a].to_datetime64())
        p1, p2, p2b = 0.5 * m2 + 0.5 * gc, 0.5 * m2 + 0.5 * g, 0.5 * m2 + 0.5 * gi; p4 = 0.4 * mL + 0.6 * gci
        for nm, s in (('MC (54 runs)', m2), ('GBM BTC only', g), ('GBM BTC+coins', gc), ('GBM BTC+intrabar', gi), ('pick 2 blend', p2),
                      ('pick 1 blend', p1), ('pick 2b blend', p2b), ('MCL (90 runs)', mL), ('GBM coins+ib', gci), ('pick 4 blend', p4)):
            x, f = xfold_ap(s), fold_ap(s[OFF:])
            print(f'{nm:16s} XF {np.round(x, 4)} mean {x.mean():.4f} | FOLDS {np.round(f, 4)} mean {f.mean():.4f}')
        for nm, s, r in (('pick 1 - MC', p1, m2), ('pick 2 - MC', p2, m2), ('pick 1 - pick 2', p1, p2), ('pick 2b - pick 2', p2b, p2),
                         ('pick 2b - pick 1', p2b, p1), ('pick 4 - pick 1', p4, p1)):
            bs = xf_bootstrap(s, r); print(f'{nm:16s} on XF: day-bootstrap mean {bs.mean():+.4f}, 90% CI [{np.percentile(bs, 5):+.4f}, {np.percentile(bs, 95):+.4f}]')
    print(f'TOTAL {time.time() - T_START:.0f}s')
