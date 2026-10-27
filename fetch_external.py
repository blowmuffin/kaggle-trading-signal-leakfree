"""External data (rules 2.6: free + public; must be declared on the forum before the deadline).
Binance spot BTCUSDT 1h klines from data.binance.vision, the exact source of the competition prices (close, indexed by the kline OPEN time).
Zips are processed in memory and SHA256-checked against Binance's .CHECKSUM files. ONLY bars before the test start (2026-06-11 20:00 UTC)
are kept, so nothing from the test period reaches disk. The closes must equal btc_train.csv exactly on the train period.
Writes external/btcusdt_1h.csv (datetime, open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote).
python fetch_external.py ETHUSDT  -> external/ethusdt_1h.csv from 2019-01 (same cut and checks; other coins are training rows only)."""
import hashlib, io, os, sys, urllib.error, urllib.request, zipfile
import numpy as np, pandas as pd
from common import D, tr, te

SYM = sys.argv[1].upper() if len(sys.argv) > 1 else 'BTCUSDT'; BTC = SYM == 'BTCUSDT'
URL = f'https://data.binance.vision/data/spot/monthly/klines/{SYM}/1h/{SYM}-1h-' + '{}.zip'
COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
CUT = pd.Timestamp(te.datetime.iloc[0])     # test start: nothing at or after it is kept


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r: return r.read()


parts = []
for m in pd.period_range('2017-08' if BTC else '2019-01', '2026-06', freq='M'):
    try: b = get(URL.format(m))
    except urllib.error.HTTPError as e:
        if e.code == 404 and not parts: continue                                  # coin not listed yet
        raise
    h = get(URL.format(m) + '.CHECKSUM').decode().split()[0]
    assert hashlib.sha256(b).hexdigest() == h, f'checksum mismatch {m}'
    with zipfile.ZipFile(io.BytesIO(b)) as z: raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS, dtype=str)
    parts.append(df[pd.to_numeric(df.open_time, errors='coerce').notna()])      # drop a header row if a file has one
k = pd.concat(parts, ignore_index=True)
ot = k.open_time.astype('int64'); ot = np.where(ot > 10 ** 14, ot // 1000, ot)  # Binance spot files switched to microseconds in 2025
k['datetime'] = pd.to_datetime(ot, unit='ms')
num = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
k = k[k.datetime < CUT][['datetime'] + num].astype({c: float for c in num}).sort_values('datetime').reset_index(drop=True)
assert k.datetime.is_unique and k.datetime.max() < CUT

if BTC:
    c = k.set_index('datetime').close.reindex(pd.to_datetime(tr.datetime))
    assert (c.values == tr.prices.values).all(), 'external closes do not equal btc_train.csv'
full = pd.date_range(k.datetime.min(), k.datetime.max(), freq='h'); miss = full.difference(k.datetime)
gaps = pd.Series(miss).diff().ne(pd.Timedelta('1h')).cumsum().value_counts().sort_values(ascending=False)
os.makedirs(D + 'external', exist_ok=True)
k.to_csv(D + f'external/{SYM.lower()}_1h.csv', index=False, encoding='utf-8', date_format='%Y-%m-%d %H:%M:%S')
print(f'{len(k)} bars {k.datetime.min()} -> {k.datetime.max()} (train: {len(tr)}, pre-train: {(k.datetime < pd.Timestamp(tr.datetime.iloc[0])).sum()})')
print(f'{"closes == btc_train.csv on all train bars; " if BTC else ""}missing hours: {len(miss)} in {len(gaps)} gaps (largest {gaps.max() if len(gaps) else 0}h); '
      f'missing inside train period: {int((miss >= pd.Timestamp(tr.datetime.iloc[0])).sum())}')
print(f'wrote external/{SYM.lower()}_1h.csv; nothing at or after', CUT, 'kept')
