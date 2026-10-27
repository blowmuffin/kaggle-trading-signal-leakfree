"""Intrabar features from Binance spot BTCUSDT 1m klines (10-22 idea search; free + public, declare on the forum if a pick uses it).
Monthly zips from data.binance.vision, SHA256-checked, processed in memory; only hourly summaries reach disk. The default run uses ONLY
minutes before the test start; --test uses ONLY test-period minutes and writes them to a separate file. Per hour T (minutes with open time in [T, T+60), all closed by T+1h, the hour's own close):
  m_ret5 / m_ret15 = log return of the last 5 / 15 minutes into the hour's close;
  m_lowage = (59 - minute of the hour's lowest 1m close) / 59  (0 = the hour closes at its low);
  m_tbs5 = taker-buy share of the last 5 minutes' volume - 0.5;  m_vsh5 = last 5 minutes' share of the hour's volume.
python fetch_m1.py [SYMBOL]  -> external/m1feat_<symbol>.csv, minutes before the test start only (~5 min per symbol, network).
python fetch_m1.py --test    -> external/m1feat_btcusdt_test.csv, BTC test-period minutes only (user decision 10-20: test-period
                                external data allowed when causal and declared); its last-minute closes must equal btc_test.csv."""
import hashlib, io, sys, time, urllib.error, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from common import D, te

CUT = pd.Timestamp(te.datetime.iloc[0])
TEST = '--test' in sys.argv; SYM = next((a.upper() for a in sys.argv[1:] if not a.startswith('--')), 'BTCUSDT')
assert not (TEST and SYM != 'BTCUSDT'), 'test-period minutes are only needed for BTC'
URL = f'https://data.binance.vision/data/spot/monthly/klines/{SYM}/1m/{SYM}-1m-' + '{}.zip'


def get(u, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(u, timeout=120) as r: return r.read()
        except urllib.error.HTTPError: raise
        except Exception:
            if k == tries - 1: raise
            time.sleep(3 * (k + 1))


def month(m):
    try: b = get(URL.format(m))
    except urllib.error.HTTPError as e:
        if e.code == 404: return None                                         # not listed yet / month not published
        raise
    h = get(URL.format(m) + '.CHECKSUM').decode().split()[0]
    assert hashlib.sha256(b).hexdigest() == h, f'checksum mismatch {m}'
    with zipfile.ZipFile(io.BytesIO(b)) as z: a = np.loadtxt(io.BytesIO(z.read(z.namelist()[0])), delimiter=',', usecols=(0, 1, 4, 5, 9), ndmin=2)
    ot = a[:, 0].astype('int64'); ot = np.where(ot > 10 ** 14, ot // 1000, ot)  # microseconds from 2025
    dt = pd.to_datetime(ot, unit='ms'); keep = (dt >= CUT) if TEST else (dt < CUT)   # history and test minutes never mix
    dt, o, c, v, tb = dt[keep], a[keep, 1], a[keep, 2], a[keep, 3], a[keep, 4]
    hrs, hi = np.unique(dt.floor('h').values, return_inverse=True); mi = dt.minute.values
    C = np.full((len(hrs), 60), np.nan); O = np.full((len(hrs), 60), np.nan); V = np.zeros((len(hrs), 60)); T = np.zeros((len(hrs), 60))
    C[hi, mi] = c; O[hi, mi] = o; V[hi, mi] = v; T[hi, mi] = tb
    C = pd.DataFrame(C).ffill(axis=1).bfill(axis=1).values                     # gaps inside the hour only
    O0 = pd.DataFrame(O).bfill(axis=1).values[:, 0]                             # the hour's first open
    d2 = np.diff(np.log(C), axis=1) ** 2                                       # column k-1 = minute k's squared 1m return
    rv = np.log(C[:, 0] / O0) ** 2 + d2.sum(1)                                 # realized variance from the hour's own minutes
    newlow = C[:, 1:] < np.minimum.accumulate(C, axis=1)[:, :-1]               # minute k prints a new low of the hour
    with np.errstate(divide='ignore', invalid='ignore'):
        F = pd.DataFrame({'datetime': hrs, 'm_close': C[:, 59], 'm_ret5': np.log(C[:, 59] / C[:, 54]), 'm_ret15': np.log(C[:, 59] / C[:, 44]),
                          'm_lowage': (59 - np.nanargmin(C, axis=1)) / 59, 'm_tbs5': T[:, 55:].sum(1) / V[:, 55:].sum(1) - 0.5,
                          'm_vsh5': V[:, 55:].sum(1) / V.sum(1), 'm_rv': rv,
                          # second generation (10-22, R22): the last 15 minutes in more detail
                          'm_ret1': np.log(C[:, 59] / C[:, 58]), 'm_vsh15': V[:, 45:].sum(1) / V.sum(1),
                          'm_tbs15': T[:, 45:].sum(1) / V[:, 45:].sum(1) - 0.5, 'm_rvsh15': d2[:, 44:].sum(1) / d2.sum(1),
                          'm_nlow15': newlow[:, 44:].sum(1)})
    return F


if __name__ == '__main__':
    months = pd.period_range('2026-06', '2026-10', freq='M') if TEST else pd.period_range('2019-01', '2026-06', freq='M')
    with ThreadPoolExecutor(6) as ex: F = pd.concat([f for f in ex.map(month, months) if f is not None], ignore_index=True).sort_values('datetime')
    assert F.datetime.is_unique and ((F.datetime >= CUT).all() if TEST else (F.datetime < CUT).all())
    if TEST:
        F = F[F.datetime <= pd.Timestamp(te.datetime.iloc[-1])]; h = pd.Series(te.prices.values, index=pd.to_datetime(te.datetime))
    else: h = pd.read_csv(D + f'external/{SYM.lower()}_1h.csv', parse_dates=['datetime']).set_index('datetime').close
    m = F.set_index('datetime').m_close.reindex(h.index).dropna(); eq = (m == h.reindex(m.index)).mean()
    assert eq > 0.999 and (not TEST or len(m) == len(h)), f'1m last close == 1h close on only {eq:.4f} of {len(m)} hours'
    # an hour whose last minute != the official close is not verifiably the same bar (e.g. the host's last test bar was taken
    # mid-hour): give it no intrabar features rather than minutes the official close may not include
    bad = F.datetime.isin(m.index[m.values != h.reindex(m.index).values])
    F.loc[bad, [c for c in F.columns if c.startswith('m_') and c != 'm_close']] = np.nan
    print(f'{int(bad.sum())} hour(s) without intrabar features (last minute != official close): {list(F.datetime[bad].astype(str))}')
    out = f'external/m1feat_{SYM.lower()}{"_test" if TEST else ""}.csv'
    F.drop(columns='m_close').to_csv(D + out, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f'{out}: {len(F)} hours {F.datetime.iloc[0]} -> {F.datetime.iloc[-1]}; last 1m close == 1h close on {eq:.4%}')
