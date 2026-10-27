# Trading Signal Classification: leak-free solution

## Summary

`Signal` is a ZigZag pivot low (1% reversal) on hourly BTC close prices. Confirming a pivot needs future prices, so the label can't be computed at decision time. We predict, at each bar *t* and from prices up to *t* only, the **probability that bar *t* will turn out to be a confirmed low**.

**The two final submissions:**

| Final pick | File (Kaggle sub) | Method | Validation AP | Public LB |
|---|---|---|---|---|
| **Pick 4** | `submission_p1star.csv` (57640210) | 0.4 × MC with longer return pools + 0.6 × gradient boosting trained on BTC + 5 other coins' history, every row with its own asset's 5 intrabar features (see *Model 4*) | XFOLDS 0.3392, FOLDS 0.3331; +0.0051 over pick 1 on 10 fresh coins | 0.29626 |
| **Pick 2b** | `submission_blend_mc_gbm_ib.csv` (57530334) | 0.5 × MC p0 + 0.5 × gradient boosting trained on BTC history, with 5 intrabar features of each bar's own 1-minute klines (see *Model 2b*) | XFOLDS 0.3335, FOLDS 0.3335 | 0.30403 |

**Earlier submissions** (not final picks; also rebuilt by `solution.py`, and the building blocks of the final two):

| Pick | File (Kaggle sub) | Method | Validation AP | Public LB |
|---|---|---|---|---|
| 1 | `submission_blend_mc_coins.csv` (57156500) | 0.5 × MC p0 + 0.5 × gradient boosting trained on BTC + 5 other coins' history | XFOLDS 0.3315, FOLDS 0.3347 | 0.30663 |
| 2 | `submission_blend_mc_gbm.csv` (57007260) | 0.5 × MC p0 + 0.5 × the same gradient boosting trained on BTC history only | XFOLDS 0.3226, FOLDS 0.3329 | 0.28800 |
| 3 | `submission_avg_p1_p2b.csv` (57592056) | 0.5 × pick 1 + 0.5 × pick 2b | XFOLDS 0.3368, FOLDS 0.3534 | 0.31004 |

FOLDS = 5 expanding time folds on the competition train set (114 lows). XFOLDS = 8 half-year blocks of Binance history, 2022H1–2025Q3 (1,173 lows). All five submissions are rebuilt exactly by `python solution.py` (about 1 minute). Every score at bar *t* depends on data up to bar *t*'s close only, and the pipeline checks this on every run (see *No-look-ahead guarantees*).

**Why this pair.** Only the better of the two final picks counts, so the pair was chosen by its expected better-of-two AP on test-sized random halves of validation data: pick 4 + pick 2b scores 0.371 on BTC history and 0.402 on 10 coins that no pick trains on, the highest of the pairs we compared (pick 1 + pick 2b: 0.369 / 0.399; pick 3 + pick 2b: 0.364 / 0.393). The two rank test `cand` bars differently enough to be worth two tickets (Spearman 0.79). The public scores were not used to choose: on validation, public and private differences between such similar files are uncorrelated.

## Data and labels

- `btc_train.csv`: 6,132 hourly rows (2025-09-29 08:00 → 2026-06-11 19:00). `btc_test.csv`: 2,628 rows (2026-06-11 20:00 → 2026-09-29 07:00), contiguous, no gaps. The prices are the **Binance spot BTCUSDT 1h kline close**, with each row labelled by the kline's open time: all 6,132 train closes match exactly.
- **`Signal` is exactly the ZigZag pivot low on closes with a 1% reversal.** The swing state starts at bar 0, and a low is confirmed once the price later rises ≥ 1% above it. This reproduces 222/222 train labels with 0 false positives; thresholds of 0.99% and 1.01% both fail.
- Confirmation needs future prices, so evaluating the rule on test prices would be look-ahead. **The pipeline never does:** every ZigZag call that could record lows on the test period uses `label_until`, so no low confirmed at or after the test start is ever computed.

## Leak-free formulation

A causal version of the same state machine runs forward one bar at a time (`TH = 0.01`, `common.zz`):

```
mode = UNDECIDED; ext = 0                       # ext = index of the running extreme of the current swing
for i in 1..N-1:
    if mode == UNDECIDED:
        if   p[i] >= p[ext]*(1+TH): mode = UP;   ext = i
        elif p[i] <= p[ext]*(1-TH): mode = DOWN; ext = i
        elif p[i] <  p[ext]:        ext = i
    elif mode == UP:
        if   p[i] > p[ext]:         ext = i
        elif p[i] <= p[ext]*(1-TH): mode = DOWN; ext = i
    else:  # DOWN
        if   p[i] < p[ext]:         ext = i
        elif p[i] >= p[ext]*(1+TH): mode = UP;   ext = i      # ext was a low, confirmed now
    cand[i] = (mode == DOWN and ext == i)                     # bar i is the running low of a down-swing
```

**`cand` is the necessary condition.** A pivot low is the running low of a down-swing at the moment it's set. Bars that aren't `cand` can't become pivots, and on train `cand` covers 221 of the 222 positives on 970 bars. Both models score only `cand` bars and give 0 elsewhere.

**What decides a `cand` bar.** Bar *t* becomes a confirmed low iff, after *t*, the price closes at or above 1.01 · p_t **before** it closes below p_t. It's a first-passage race between two barriers, and the lower one sits at the current price. That makes the outcome mostly a function of short-term volatility and luck, which is why the achievable AP is around 0.3.

## Model 1: MC p0 (in every pick)

A filtered historical simulation estimates that first-passage probability directly, with nothing fitted to the labels:

1. **Volatility:** an EWMA scale forecast `sig[t]` with half-life HL, from 1-bar log returns up to *t*.
2. **Return pool:** the trailing W returns, each standardised by the scale known *before* it (`z_s = r_s / sig[s-1]`) and demeaned, so there's zero drift.
3. **Paths:** N = 1,000 paths of H = 72 steps from p_t. Each step is `sig[t] · z`, with z drawn in blocks of L = 3 consecutive pool values.
4. **p0[t]** = the fraction of paths that close ≥ 1.01 · p_t before any close < p_t. For a `cand` bar this equals running the full ZigZag forward from the actual state; we checked that bit-exact.
5. **Ensemble:** the average over W ∈ {120, 240, 480} × HL ∈ {6, 12, 24} × seeds 0–5 (54 runs), run on the Binance-history + competition series. Pick 4 adds 36 runs with W ∈ {960, 1920} (90 runs, "MCL").

Each bar's random numbers are seeded with `seed + t`, so a bar's value doesn't depend on the series length. The MC is calibrated: on validation `cand` bars its mean p0 is 0.226 against an actual rate of 0.225. Block bootstrap (L = 3) and iid resampling score the same on the 1,173-low history (+0.0016, 90% CI [−0.003, +0.006]).

## Model 2: history-trained gradient boosting (the GBM in every pick)

- **Training data:** Binance BTCUSDT 1h closes from 2019-01-01 to the train end (≈ 59k bars, ~2,600 lows), plus the competition train set. Labels are the ZigZag lows of this continuous series. They equal the official `Signal` on train except train row 4, a low only because the official ZigZag restarts at row 0.
- **Features (17, all causal, `common.feats`):** `cand`; the swing state `dd` = log(p_t / swing high), `hrs_from_hi`, and `nlows` = new lows in the current down-swing; returns over 1, 2, 3, 6, 12 and 24 bars; rolling std of 1-bar returns over 6, 24 and 72 bars; RSI(14) scaled 0–1; log price minus its 24-bar mean; hour and day of week.
- **Purged labels:** a `cand` row is used for a fit starting at bar *a* only if its fate was already decided before *a*, i.e. the first later close below p_s or at/above 1.01 p_s came before *a*. Non-`cand` rows are never lows, so their label is known immediately.
- **Model:** `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=0)`. Early stopping is off because sklearn's default turns it on above 10k rows with a random holdout, which made fits unstable. The prediction is multiplied by `cand`.

- **Pick 1's GBM adds other assets as training rows.** The same 17 features and labels are built from Binance 1h ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT and ADAUSDT closes, 2019-01 (SOL from its 2020-08 listing) to the train end, about 312k rows. A coin row enters a fit starting at BTC bar *a* only if it is timestamped before *a* and its label was resolved before *a*. Test predictions are made for BTC from the competition prices only.

The GBM was data-starved, not feature-starved: trained on competition data alone it scores FOLDS 0.279; on 2019+ BTC history it matches the MC; and the cross-asset rows lift it again (see Validation). More BTC history alone had flattened out, so it's the variety of assets, not the row count, that helped.

## Model 2b: intrabar features (pick 2b, 2026-10-22)

The BTC-only gradient boosting of pick 2 gets **5 more features per hourly bar, built only from that bar's own 60 one-minute klines** (Binance BTCUSDT 1m, `fetch_m1.py`), all of which close by the bar's close:
- `m_ret5` and `m_ret15`: the return of the last 5 and 15 minutes into the close.
- `m_lowage`: minutes since the hour's lowest 1m close (0 means the hour closes at its low).
- `m_tbs5`: the taker-buy share of the last 5 minutes' volume.
- `m_vsh5`: the last 5 minutes' share of the hour's volume.

They say whether a fresh low is still being sold into at the close or is already turning. On 9,888 history `cand` bars, the univariate AUC is 0.537 for `m_vsh5` (a volume burst at the close makes the bounce likelier) and 0.45 for `m_ret15` and `m_lowage` (falling into the close makes it less likely).

**Alignment.** Each hour's last-minute close must equal the official hourly close. That holds on 100% of history hours and on 2,627 of the 2,628 test hours. The exception is the final test hour (2026-09-29 07:00): the competition's price there was evidently taken before that hour ended, so the hour gets no intrabar features (it's a non-`cand` bar, scored 0).

**Validation.** Blended 0.5 / 0.5 with the MC as above, it scores XFOLDS 0.3335 vs pick 2's 0.3226: +0.0109, better on 7/8 blocks, paired day-bootstrap 90% CI [+0.0012, +0.0203]. FOLDS is +0.0006.

**Leak checks.** An independent reviewer recomputed the features by hand from a raw Binance zip (matching to 1e-16). The features correlate with their own hour's return (+0.19 to +0.31), not the next hour's (−0.07 to +0.05), just like the honest `ret1`. As a positive control, the same features shifted by one hour, a deliberate leak, would score +0.216 AP. Shuffled across rows they lose 0.004; taken from the previous hour they lose 0.001. So the gain comes from bar *t*'s own minutes and nothing later.

## Model 4: the cross-asset GBM with every asset's own intrabar features (pick 4, 2026-10-25)

- **GBM:** pick 1's training rows (BTC + ETH/BNB/SOL/XRP/ADA, same purge), with the 5 intrabar features of Model 2b on **every** row, each from its own asset's 1m klines (`fetch_m1.py ETHUSDT` etc., history only; BTC test bars use `m1feat_btcusdt_test.csv` as in pick 2b).
- **MC (MCL):** the 54 runs plus 36 on longer trailing pools, W ∈ {960, 1920} × HL ∈ {6, 12, 24} × 6 seeds (90 runs).
- **Blend:** `0.4 · MCL + 0.6 · GBM`, per bar.
- **How it was chosen.** 19 variants were screened on BTC XFOLDS and on the 5 coins as targets (`probe_r5.py`, `probe_r6.py`; none passed alone). The three best were combined and pre-registered in `probe_r7.py`, then judged only on 10 coins never used to choose them (DOGE, LTC, LINK, TRX, BCH, ETC, XLM, ATOM, DOT, AVAX; 80 coin-blocks over the same 8 calendar blocks; never training rows). Result vs pick 1: **+0.0051 AP, better in 55/80 coin-blocks** (bootstrap P(≤0) 0.00; 0.001 clustered by calendar block, 0.004 by coin), better on 8 of 10 coins. Most of the gain is the coins' own intrabar rows (+0.0042 alone). On BTC XFOLDS (the screening data, so not a confirmation): +0.0077, 5/8 blocks, day-bootstrap 90% CI [−0.0014, +0.0170]; FOLDS −0.0016.
- **Pair.** With pick 2b, the expected better-of-two private AP rises by +0.003 (BTC history halves and the fresh coins alike). It ranks test `cand` bars 0.945 alike with pick 1 and 0.79 with pick 2b.
- **Checks.** Truncation test for the W = 1920 MC; coin intrabar files end before the test start (asserted); the `--audit` rebuild at 6 test bars covers pick 4; an independent reviewer rebuilt it through the research code and matched the file to 8e-17.

## Blends (every pick)

Each pick is a **per-bar probability average** of the MC and one GBM: `0.4 · MCL + 0.6 · GBM` for pick 4 (cross-asset GBM with intrabar features), and `0.5 · MC p0 + 0.5 · GBM` for picks 2b (BTC GBM with intrabar features), 1 (cross-asset GBM) and 2 (BTC GBM). Both components are probabilities of the same event on `cand` bars, so bar *t*'s blended score depends on bar *t* only. We deliberately don't rank within the test set: a rank there would make each bar depend on every other bar's score, including future ones.

The two models are about equally strong but rank `cand` bars differently (Spearman 0.63 on history, 0.68 on validation, 0.52 on test), which is why they blend.

## Interpretability (`python explain.py`, validation data only)

**What drives a prediction.** A `cand` bar becomes a low iff the price rises 1% before closing below it, a race whose odds are set mainly by how far the price moves per hour relative to the 1% barrier. Both models learn exactly this.

- **MC p0 is a volatility ranker.** On the 4,507 history `cand` bars of the 8 XFOLDS blocks, Spearman(p0, EWMA 1-hour volatility / 1%) = **0.905**. By p0 decile, the actual low rate climbs from 0.164 (vol 0.28 × barrier) to 0.344 (vol 1.24 × barrier): the more the market moves per hour, the likelier a fresh low gets its 1% bounce before being broken. The decile table is in `explain_mc_deciles.csv`.
- **History GBM: out-of-sample permutation importance** (drop in block AP when the feature is shuffled within the block, model refitted at each block start; 8 blocks × 3 shuffles; `explain_gbm_importance.csv`):

| Feature | Importance | Blocks > 0 | Meaning |
|---|---|---|---|
| `cand` | 0.040 | 6/8 | the bar is the running low of a down-swing (the necessary condition) |
| `vol72` | 0.016 | 7/8 | 72-hour return volatility: the volatility regime |
| `ret2` | 0.015 | 7/8 | 2-hour return: how sharp the last drop was |
| `dd` | 0.010 | 5/8 | drawdown from the swing high |
| `vol24`, `vol6` | 0.006, 0.005 | 4/8, 5/8 | shorter-term volatility |
| `ret3`, `ret24`, `hour`, `hrs_from_hi`, `ret12`, `ret1` | 0.002–0.005 | 3–5/8 | minor |
| `rsi`, `ret6`, `nlows`, `dow`, `dist_sma24` | ≤ 0 | — | no usable signal |

Calendar effects (`hour`, `dow`) and trend indicators (RSI, distance to the moving average) carry essentially nothing. Volatility, the swing state and the speed of the last move carry the signal.

## External data

- **Source:** Binance spot 1h and 1m klines for BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT and ADAUSDT, from the free public dump https://data.binance.vision/?prefix=data/spot/monthly/klines/ (no account needed). It's declared on the competition forum as rule 2.6b requires.
- **Retrieval:** `fetch_external.py [SYMBOL]` downloads the monthly zips (BTC from 2017-08, the coins from 2019-01, up to 2026-06) in memory and verifies each one's SHA256 against Binance's `.CHECKSUM` file. It keeps **only bars before the test start (2026-06-11 20:00 UTC)** and, for BTC, asserts the closes equal `btc_train.csv` exactly. Picks 1 and 2 use no test-period external data.
- **Picks 2b and 4: BTC 1m klines, test period included.** `fetch_m1.py` reads Binance BTCUSDT 1m zips with the same checksums, and writes only hourly summaries of each hour's own minutes. History goes to `external/m1feat_btcusdt.csv`, and the test period to a separate `external/m1feat_btcusdt_test.csv`, whose last-minute closes must equal `btc_test.csv`. A test bar's features use only minutes that close by that bar's close, the same moment its competition price is taken. This is causal, and declared on the forum like the rest.
- **Pick 4 also: the 5 coins' 1m klines, history only.** `fetch_m1.py ETHUSDT` (etc.) writes `external/m1feat_<coin>.csv` with the same hourly summaries, minutes before the test start only (asserted), used only as training-row features.
- **Use:** training data and validation (XFOLDS) for the GBMs (the coins only as training rows, for picks 1 and 4), and the pre-train part of the series the MC runs on. Test-time inputs are the competition's test prices, plus, for picks 2b and 4, each test bar's 5 intrabar summaries from `external/m1feat_btcusdt_test.csv` (that bar's own minutes only). Picks 1 and 2 use the test prices alone.

## Validation

- **FOLDS:** 5 expanding windows on train, never shuffled. Fold *k* trains on `[0, a)` and validates on `[a, b)`, with `a = int(6132·(0.5+0.1k))` and `b = int(6132·(0.6+0.1k))`. They hold 35/25/19/11/24 lows, so they're noisy.
- **XFOLDS:** 8 half-year blocks of Binance history, 2022H1 … 2025-07 → train start, holding 1,173 lows. MC needs no fitting. The GBM is refitted at each block start on purged rows from 2019 on. XFOLDS was not used to choose features or hyperparameters, so it's the less optimistic yardstick.

| Model | XFOLDS mean (8 blocks) | FOLDS mean (5 folds) |
|---|---|---|
| MC p0, 54 runs (picks 1, 2, 2b) | 0.3154 | 0.3154 |
| MCL, 90 runs (pick 4) | 0.3190 | 0.3040 |
| GBM, BTC history only (pick 2) | 0.3148 | 0.3238 |
| GBM, BTC history + intrabar (pick 2b) | 0.3294 | 0.3298 |
| GBM, BTC + 5 coins (pick 1) | 0.3309 | 0.3225 |
| GBM, BTC + 5 coins, every row with own intrabar (pick 4) | 0.3363 | 0.3194 |
| Pick 2: 0.5 · MC + 0.5 · BTC GBM | 0.3226 (beats MC in 5/8 blocks) | 0.3329 |
| Pick 1: 0.5 · MC + 0.5 · cross-asset GBM | 0.3315 (beats MC in 8/8 blocks) | 0.3347 |
| **Pick 2b (final)**: 0.5 · MC + 0.5 · BTC + intrabar GBM | **0.3335** (beats MC in 7/8 blocks) | **0.3335** |
| **Pick 4 (final)**: 0.4 · MCL + 0.6 · cross-asset + intrabar GBM | **0.3392** (beats MC in 7/8 blocks) | **0.3331** |
| Reference: the same GBM trained on competition data only | — | 0.2787 |

Per-block XFOLDS AP (2022H1 … 2025Q3):
- MC: 0.324, 0.255, 0.367, 0.297, 0.351, 0.340, 0.323, 0.269
- pick 2: 0.310, 0.289, 0.375, 0.308, 0.380, 0.327, 0.342, 0.250
- pick 1: 0.331, 0.260, 0.412, 0.301, 0.376, 0.342, 0.349, 0.281
- pick 2b: 0.331, 0.273, 0.375, 0.325, 0.397, 0.329, 0.361, 0.278
- pick 4: 0.323, 0.257, 0.429, 0.317, 0.392, 0.342, 0.355, 0.299

Per-fold FOLDS AP:
- pick 1: 0.376, 0.290, 0.269, 0.286, 0.453
- pick 2: 0.365, 0.303, 0.293, 0.233, 0.471
- pick 2b: 0.364, 0.285, 0.336, 0.274, 0.408
- pick 4: 0.375, 0.271, 0.305, 0.308, 0.408

Paired bootstrap on XFOLDS, resampling calendar days within each block:
- pick 1 − MC: +0.0151, 90% CI [+0.0044, +0.0254]
- pick 2 − MC: +0.0072, 90% CI [−0.0061, +0.0201]
- pick 1 − pick 2: +0.0079, 90% CI [−0.0031, +0.0190]
- pick 2b − pick 2: +0.0108, 90% CI [+0.0012, +0.0203]
- pick 4 − pick 1: +0.0077, 90% CI [−0.0014, +0.0170] (on the data its components were screened on; its confirmation is the 10 fresh coins, see *Model 4*)

Pick 4 is the strongest model. Pick 2b ranks test bars differently (Spearman 0.79 with pick 4 on test `cand` bars) and is the diversified second ticket.

**Tried and rejected** (each with a pre-set acceptance rule):
- hour-of-day volatility in the MC; analog paths; path-updated volatility;
- stacking on `cand` bars;
- extra GBM features;
- Binance OHLCV features (range volatility, wicks, volume and trade-count z-scores, taker-buy share): +0.002 XF;
- a public notebook's 52 technical indicators (RSI family, Bollinger, MACD, etc.): ≤ +0.002 XF;
- more BTC history alone (from 2017): +0.003 XF;
- rank-based blends, which are non-causal.

The full ledger is in `notes/ledger.csv`.

## No-look-ahead guarantees (checked on every run of `solution.py`)

- **Truncation tests.** The MC p0 at the first and last test `cand` bars is recomputed from the truncated series `p[:t+1]`, including its own volatility forecast and return pool (W = 240 and pick 4's longest pool W = 1920), and must be identical. The 17 features at the train end, the test start and the last bar (and on each coin series) are recomputed from truncated data and must match.
- **Intrabar features.** Each hourly row summarises only that hour's own 1m klines, all closed by the hour's close (`fetch_m1.py` asserts each hour's last-minute close equals the official close; the one test hour where it doesn't gets no features). History and test-period minutes are written to separate files, and the coins' files end before the test start (asserted).
- **No test labels.** `zz(..., label_until=test start)` means no low confirmed on test prices is ever recorded. The script asserts that the label vector is all zero on the test period.
- **Purging.** Every training row of every fit, BTC or coin, is timestamped and has its label resolved before the fit start (asserted). The final models are fitted on rows before the test start only, and every coin file ends before the test start (asserted).
- **Causal blend.** It's a per-bar average of two causal scores.
- **Independent review.** A separate reviewer with fresh context audited each pipeline. For the BTC GBM it recomputed test scores at four test bars from `xp[:t+1]` only, refitting from pre-test data, and they matched to 1e-15. For the cross-asset GBM it checked every fit window (no coin row timestamped or resolved at or after the fit start), re-derived coin labels from pre-cut prices only (0 mismatches), checked coin/BTC time alignment, and reproduced the scores to 8e-17.
- **End-to-end audit (`python solution.py --audit`).** This is the jury's one-command check. At 6 test rows (the train/test boundary row, the first test `cand` bar, three spread through the test period and the last `cand` bar), it rebuilds the whole score from the price series cut at that row. That covers the ZigZag state, the training labels and purge times, the MC runs (54, or 90 for pick 4), the 17 features, the intrabar rows up to that row, and every GBM fit. An independent reviewer planted a one-bar look-ahead in a single feature, and the audit failed at every `cand` bar. It asserts the result equals the submitted value for every pick, both final picks included; any look-ahead would change it. Every row passes, to < 1e-12 (last run 2026-10-27).
- **Reproduction.** `solution.py` asserts every output equals its submitted file (max abs diff < 1e-12; 8e-17 for both final picks). Clean-room runs (fresh folder, fresh Binance download, bundle files only) reproduced picks 1 and 2 on 2026-10-14 and 2026-10-20, and pick 2b on 2026-10-22 including the BTC 1m history and test downloads. On 2026-10-27 the final bundle (the `README.md` commands run verbatim, in a fresh folder with a fresh virtualenv from `requirements.txt`) reproduced all five files, pick 4 included (≤ 1.9e-16), with every download made from scratch, the coins' 1m klines included. The audit passed. The re-downloaded files match the ones used for the submissions.

## Reproduce

- **Environment:** Python 3.13.7, with numpy 2.3.3, pandas 2.3.3 and scikit-learn 1.8.0 (`requirements.txt`). Tested on Windows 11, using 12 cores for the MC; any core count gives the same numbers.
- **Data:** put `btc_train.csv`, `btc_test.csv` and `sample_submission.csv` next to the scripts.

```
pip install -r requirements.txt
python fetch_external.py      # Binance BTC history -> external/btcusdt_1h.csv (~1.5 min, internet)
for s in ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT; do python fetch_external.py $s; done   # the 5 coins (~1.5 min each)
python fetch_m1.py; python fetch_m1.py --test   # BTC 1m -> intrabar features, history and test kept apart (picks 2b, 4; ~5 min)
for s in ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT; do python fetch_m1.py $s; done        # the coins' 1m history (pick 4; ~5 min each)
python solution.py            # -> final/pick4_p1star.csv, final/pick2b_blend_btc_intrabar.csv (+ picks 1, 2, 3) and all checks (~1 min)
python solution.py --cv       # also the validation tables above (~3 min)
python solution.py --audit    # also the end-to-end look-ahead audit at 6 test rows (~1 min)
```

The run is deterministic: seeded MC and fixed GBM `random_state` with early stopping off.

## Files

- `solution.py`: the final pipeline (MC p0, BTC-only and cross-asset GBMs, blends, leak tests, reproduction check).
- `common.py`: data loading, folds, causal ZigZag, features, label purging helpers.
- `fetch_external.py`: external-data download and verification.
- `fetch_m1.py`: Binance 1m klines → hourly intrabar features for picks 2b and 4 (`python fetch_m1.py [SYMBOL]` for history, `python fetch_m1.py --test` for BTC's test period, written to a separate file).
- `explain.py`: interpretability tables (MC deciles, GBM permutation importance) on validation data.
- `requirements.txt`: pinned packages.
- `submission_*.csv`: the files submitted to Kaggle; `solution.py` checks its rebuild against each one present.
