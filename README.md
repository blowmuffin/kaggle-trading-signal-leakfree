# Leak-free trading-signal pipeline (team Ayush Kasare)

Code for the Kaggle competition [Tabular Time Series Trading Signal Classification](https://www.kaggle.com/competitions/tabular-time-series-trading-signal-classification). It rebuilds both final submissions from raw data and checks that no score uses future prices.

| Kaggle submission | File | Model |
|---|---|---|
| 57640210 | `submission_p1star.csv` | 0.4 × Monte Carlo p0 (90 runs) + 0.6 × gradient boosting trained on BTC + ETH/BNB/SOL/XRP/ADA history |
| 57530334 | `submission_blend_mc_gbm_ib.csv` | 0.5 × Monte Carlo p0 (54 runs) + 0.5 × gradient boosting trained on BTC history |

Both gradient-boosting models also use 5 summaries of each hourly bar's own 1-minute klines, all closed by that bar's close. The score at bar *t* depends only on data up to bar *t*'s close. The method, validation and interpretability are in [SOLUTION.md](SOLUTION.md).

## Reproduce

Python 3.13 with the pinned packages, and internet access to https://data.binance.vision. The run takes about 20 minutes on a fast connection, mostly downloads.

1. Install the packages: `pip install -r requirements.txt`
2. Put the competition's `btc_train.csv`, `btc_test.csv` and `sample_submission.csv` in this folder.
3. Run, from this folder (bash):

```bash
python fetch_external.py
for s in ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT; do python fetch_external.py $s; done
python fetch_m1.py
python fetch_m1.py --test
for s in ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT; do python fetch_m1.py $s; done
python solution.py --audit
```

The fetch scripts write `external/`, and `solution.py` writes the rebuilt submissions to `final/`.

## What the run checks

- **Reproduction:** it rebuilds every submission and asserts it equals the submitted file next to it, printing `REPRODUCES` with the largest difference (about 1e-16).
- **Leak tests:** the simulation and the features are recomputed from the series cut at test rows (and the train/test boundary) and must match. The checks also cover training labels (used only once resolved before the fit start), the absence of test labels, and coin files that end before the test start. They end with `LEAK TESTS PASSED`.
- **End-to-end audit (`--audit`):** at 6 test rows it rebuilds the whole score from data cut at that row: the simulation, features, swing state, training labels and every model fit. The result must equal the submitted value, so any look-ahead would fail it. It ends with `END-TO-END AUDIT PASSED`.

## External data

All of it is free and public Binance spot klines from https://data.binance.vision, each monthly file SHA256-checked against Binance's checksum. It was declared in the competition forum on 2026-10-25: https://www.kaggle.com/competitions/tabular-time-series-trading-signal-classification/discussion/752661

- **1h klines** for BTCUSDT and the 5 coins: history only, all before the test start. For BTC the script asserts the closes equal `btc_train.csv`.
- **1m klines** for the 5 coins: history only, used only for training rows.
- **1m klines for BTCUSDT, test period included.** Each hourly row gets 5 summaries of that hour's own minutes, all closed by the hour's close. The test-period summaries are written to their own file, `external/m1feat_btcusdt_test.csv`.

## Files

- `solution.py`: the final pipeline, with the leak tests, reproduction check and audit.
- `common.py`: data loading, folds, the causal ZigZag, features and label-purging helpers.
- `fetch_external.py`, `fetch_m1.py`: external-data download and verification.
- `explain.py`: interpretability tables on validation data.
- `SOLUTION.md`: the full write-up.
- `submission_*.csv`: the files submitted to Kaggle. The two above are the final ones; the other three are earlier submissions that `solution.py` also rebuilds.
