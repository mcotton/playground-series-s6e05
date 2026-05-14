# Exploration Options - Playground Series S6E5

## Initial Prompt for Claude

> Hi Claude, it is a new month and that means a new Kaggle competition. We are working on `https://www.kaggle.com/competitions/playground-series-s6e5/overview` and the data is already downloaded into `@archive/`. Your job is to help me learn how I can improve and get a competitive score. Understanding and experimentation are more important than winning.
>
> Rules:
> - Do not provide code unless I explicitly ask.
> - Do not add or modify `.ipynb` files without my permission.
> - Do not perform EDA for me — guide me through it instead.
> - Everything runs inside of Docker.
> - Keep running notes about what we learned, what we tried, and what is still yet to try in this document (`exploration_options.md`, which we'll refer to as 'md').
> - **Track experiments in this document** (CV scores, LB scores, what changed) — we are not using a separate `submission_notes.ipynb` this time.
> - Ask if you need clarification.

## Competition Summary
- **Task**: Binary classification — predict `PitNextLap` (does the driver pit on the next lap?)
- **Metric**: ROC-AUC
- **Train rows**: 439,140
- **Test rows**: 188,165
- **Class balance / target distribution**: TBD (EDA)
- **Original dataset available?**: TBD

## Dataset Features (15 total + `id` + target `PitNextLap`)

### Numeric (12)
| Feature | Correlation w/ Target | Notes |
|---------|----------------------|-------|
| Year | | 2022–2025 in samples seen |
| PitStop | | Cumulative count of pit stops so far in the race (not a leak) |
| LapNumber | | |
| Stint | | Stint number within race |
| TyreLife | | Laps on current set of tyres |
| Position | | |
| LapTime (s) | | |
| LapTime_Delta | | Delta vs. some reference (TBD) |
| Cumulative_Degradation | | Pre-engineered — may be redundant w/ TyreLife |
| RaceProgress | | LapNumber / total laps, presumably |
| Position_Change | | |

### Categorical (3)
| Feature | Cardinality | Notes |
|---------|-------------|-------|
| Driver | TBD | Driver code (e.g., `D109`, `VER`) — mixed format worth checking |
| Compound | TBD | HARD / MEDIUM / SOFT / etc. |
| Race | TBD | Grand Prix name |

## Implications of ROC-AUC
- Threshold-independent — no class-weight tuning needed *for the metric itself*.
- No need to calibrate probabilities; only **rank order** matters.
- `scale_pos_weight` rarely helps AUC directly (it shifts probs, not ranks). Only worth trying if it changes which features the model splits on.
- Class imbalance hurts learning, not scoring — focus on the model finding signal in the minority class, not on rebalancing the metric.
- Submit raw probabilities, not 0/1.

## Key Observations
- **`year` dominates feature importance (0.336).** Top 3 features (`year` + `stint` + `tyrelife`) carry ~75% of total importance. Healthy distribution otherwise.
- **2023 has anomalously low pit rate (~1% in train, ~3% in original) vs ~27-30% for other years.** A ~30× difference. Almost certainly a data generation artifact inherited from the source. Explains why `year` is the #1 feature: "if year=2023, predict ~0" is a near-perfect rule.
- **Train and test year distributions are within 0.2pp** across all four years — model's year-conditional learning generalizes correctly to LB.
- **Dropping `driver` was the single biggest gain (+0.0086 LB).** High-cardinality categorical with hundreds of low-support levels was driving overfit, not signal.
- **Custom features `wet_race` and `avg_stint_per_race` were marginally noisy.** Dropping them was net-flat on LB; cleaner feature set retained.
- **Encoding `year` as categorical (Exp 9) was a clean +0.005 LB.** Year's response to target is non-monotonic (2023 anomaly), so numeric splits are inefficient. Categorical partition splits isolate 2023 in one node. Cleaner than the year_freq attempt (Exp 8), which overfit (CV up, LB down).
- **Importance ranking is not a proxy for score** (Exp 8 lesson). Year_freq + compound_freq looked promising on importance redistribution, but the actual score went the wrong direction. Always validate with CV/LB.

## CV-LB Divergence (Exp 2)
- Adding the original dataset moved CV down 0.0070 but LB up 0.0017. CV std widened from 0.010 → 0.014.
- **Best read of what's happening:** original rows are sequentially coherent, train/test rows are scrambled. Adding original to training shifts the model toward patterns that generalize slightly better to LB (which may be closer to original distribution than to noisy train) but score worse on held-out *train* rows in CV (which are still scrambled).
- **CV is now less trustworthy.** It's evaluating against scrambled rows; the model is trained on a mix; the score reflects the mix-vs-scrambled gap, not the mix-vs-test gap.
- **LB improvement may be real or noise.** +0.0017 is small relative to typical LB jitter on Kaggle. Without resubmitting at a different seed, can't be certain. But the *direction* makes sense given the cleaner original data.
- **Suggested next moves to disambiguate:**
  - Try `sample_weight` weighted toward original rows (e.g., 1.5x weight on original, 1x on train) — if LB keeps climbing, the original is providing real signal.
  - Or weighted *against* original (0.5x) — if LB stays high, the original is mostly redundant; if LB drops, the original is doing real work.
  - Re-run Exp 2 with a different random_state to test stability of the LB improvement (saves a submission if you can hold off until convinced).

## CV-LB Gap Analysis (after Exp 0)
- **CV: 0.92156, LB: 0.93671 — LB is +0.015 higher than CV.** Inverse of the usual overfit pattern.
- **Likely cause:** `StratifiedGroupKFold` by `(race, year)` holds *entire races* out of training. The model never sees that race's specific context (track-specific tyre wear, typical pit windows, traffic patterns) during validation, so CV measures generalization to *unseen races*. The Kaggle test set was almost certainly split row-wise from the same race-year pool as train, so on the LB the model sees familiar races. CV is harder than the LB scenario — exactly what we want.
- **Implication:** trust CV as a *lower bound*. Improvements on CV will translate to LB; a CV bump is real signal, not just LB noise.
- **Sanity check worth doing once:** `set(test_df.Race + test_df.Year) - set(train_df.Race + train_df.Year)` — if it's empty, test races are all seen in train (random split, validates the theory above). If non-empty, group CV is even more important and the gap may shrink.
- **Don't switch back to plain `StratifiedKFold`** chasing a higher CV — it would inflate CV without helping LB, and you'd lose the early-warning system for race-context overfitting.

## Open Questions (flagged for EDA)
- ~~**CV strategy**: data is per-(Driver, Race, Year, Lap). Random KFold will leak race/driver context across folds. Likely want **GroupKFold by `Race`** or by `(Race, Year)`.~~ **Resolved:** using StratifiedGroupKFold by `(race, year)`. Empty intersection check skipped — train/test alignment confirmed by year distribution match.
- ~~**Driver coding**: mixed real (`VER`) and anonymized (`D109`) codes.~~ **Resolved:** dropped `driver` entirely (Exp 6) — was the single biggest LB gain.
- ~~**Class balance**: pit events are rare per lap — likely 5–10% positive.~~ **Resolved:** ~20% positive overall, but heavily year-dependent (1% in 2023, ~27-30% other years).

## Plan for Rest of Month
- **Optuna deferred to end of month.** Tuning hyperparameters on a feature set that may still change wastes the search. Save Optuna for the final feature set.
- **Priority order for remaining experiments (between now and end-of-month):**
  1. **Reality-check the original blend with the cleaner feature set.** Try Exp 7 minus original concat — does the orig blend still help now that `driver` is dropped?
  2. **Model class blend (LightGBM + CatBoost averaged with XGBoost).** Standard +0.001-0.003 lever, different inductive biases.
  3. **More decisive `sample_weight` test** (5x on original) — settles whether to keep orig blend.
  4. **Pseudo-labeling** — train on confident test predictions added to train. Higher risk, +0.001-0.003 typical.
  5. **Year-conditional aggregates** — given how dominant `year` is, features like "tyrelife relative to year-typical at this lapnumber" might capture extra signal. Lower priority.
- **Then Optuna** on the locked-in feature set.

## Data Integrity Findings (2026-05-01)

### Train data is desequenced
- `stint == pitstop + 1` holds only **52.3%** of the time in `train.csv`. In real F1 (and in `f1_strategy_dataset_v4.csv` for verified drivers), this should be 100%.
- Plotting `stint` vs `lapnumber` for a single (driver, race, year) shows non-monotonic up/down jumps — physically impossible since you can't un-pit.
- **Conclusion:** the Kaggle train/test data is generated by row-wise sampling/perturbation, not preserving the underlying race sequences. Each row is essentially an independent snapshot.

### `PitStop` semantics: documentation says 0/1 per lap, data pattern is cumulative
- Original dataset documentation: `PitStop – Whether the driver pitted on that lap (0/1)`.
- But HAM/Austrian GP/2022 shows `pitstop` values of 0,0,0...,1,1,1,...,2,2,... — consecutive 1s across laps 23-28 can't be "pitted that lap" (no driver pits on 6 consecutive laps). The pattern is consistent with **cumulative count of pit stops**, possibly with a generation quirk where the increment registers a few laps before stint/tyrelife reset.
- Anomaly worth noting but not solving: `pitstop` increments at lap 23 while `stint` flips and `tyrelife` resets at lap 29 — a 6-lap discrepancy in the original data. Cause unknown (drive-through penalty? lagged indicator? synthetic generation artifact?).
- **Leak status: NO LEAK (verified).** `df.groupby('pitstop')['pitnextlap'].mean()` returned 0.191 (pitstop=0) and 0.248 (pitstop=1). A 1.3× ratio, not the dramatic spike a forward-looking leak would produce. Consistent with cumulative-count semantics — drivers who've already pitted once are slightly more likely to pit again because they're in active multi-stop strategies.
- **Decision:** keep `pitstop` as a feature. Drop the leak hypothesis.

### Class balance — much higher than expected
- Overall pit rate ≈ 20% (per `groupby('pitstop')` showing 19% baseline). Not the rare-event ~5% I initially guessed.
- Likely the dataset oversamples pit-relevant laps, or `pitnextlap` flags more than just the literal next-lap-before-pit.
- Consequence: class imbalance handling is even less of a concern than usual for AUC. Skip `scale_pos_weight` etc. without hesitation.

### Implications for feature engineering plan
- **Lap-context (sequence) features are largely DEAD ON ARRIVAL** for the desequenced rows. `tyrelife_diff` between two rows that aren't actually adjacent laps is garbage.
- **Salvage path tested and rejected:** corruption is **uniform** across real and anonymized drivers (52.4% coherence for non-anon, 52.9% for anon — both desequenced).
  - Filtering to "real" drivers won't recover sequence.
  - `is_anon_driver` is not a useful feature — both subsets behave the same.
  - Lap-context features confirmed dead for the entire dataset.
- **What still works regardless of sequence corruption:**
  - Native categorical handling of `compound`, `race`, `driver`.
  - Per-row features (no lap-context dependency).
  - Cross-row aggregates that don't assume sequence (per-driver pit rate, per-race pit lap distribution).
  - Optuna tuning, original-dataset blend (carefully — see below).
- **Original dataset blend is now more interesting**: if train is desequenced and original is clean, blending in clean rows could provide signal the synthetic rows don't.

## Current State
- Pipeline in `common.py`, model code in `xgboost.ipynb`
- No model or CV pipeline set up yet

---

## Things to Try

### Baseline (Priority: High)
- [x] Get a baseline XGBoost model working
- [x] Set up stratified CV (or KFold for regression) with the **competition metric** as scoring
- [x] Establish baseline CV score — 0.92156 ± 0.00995
- [x] Submit baseline to confirm CV-LB correlation — LB 0.93671 (CV pessimistic by +0.015, healthy gap)

### Class Imbalance (mostly N/A for AUC)
- [x] ~~`sample_weight='balanced'`~~ — doesn't help AUC, dropped from notebook
- [ ] ~~`scale_pos_weight`~~ — same reasoning, skip unless seeing very low minority recall in OOF preds
- [ ] ~~Threshold tuning~~ — AUC is threshold-independent
- [ ] **OOF minority recall sanity check**: are the positive predictions actually finding the pit laps, or is the model just ranking obvious negatives well? If most CV AUC comes from "easy negatives," there's room.

### Feature Engineering
- [ ] Interactions and ratios that trees can't find via rectangular splits
- [ ] Group-based aggregations (e.g., mean numeric per categorical)
- [ ] Target encoding (with proper CV fold separation to avoid leakage)
- [ ] Skip pre-engineered booleans / hand-crafted formulas — XGBoost finds these itself

#### Lap-Context (Sequence) Features — DEAD ON ARRIVAL (see Data Integrity Findings)
~~The model currently has **zero memory across rows**. Each lap is treated independently, but pit decisions depend on what just happened. These features require row-to-row comparison within a `(driver, race, year)` sequence — info trees cannot derive from rectangular splits.~~

**Don't try these. The train data is desequenced uniformly — lap-to-lap diffs and rolling stats compute over rows that aren't actually adjacent in any race.** Notes preserved below for historical reference only.

**Implementation gotchas:**
- **Sort first**: `df.sort_values(['driver', 'race', 'year', 'lapnumber'])` before any `groupby().shift()` or `.diff()`.
- **Compute on `pd.concat([train, test])`** (not separately). Train and test rows from the same (driver, race, year) sequence interleave (Kaggle did a random row split), so computing on train alone leaves test rows with NaN previous-lap features. Combined-frame computation is **not leakage** for sequence features — the lag/diff only uses earlier laps in the same race, which is information available at prediction time.
- **First lap of each group will be NaN** — fill with 0 (no prior lap to diff against) or leave as NaN and let XGBoost handle it via missing-value splits.

**Specific features to try (cheap → expensive):**
- [ ] `tyrelife_diff` — `groupby(['driver','race','year'])['tyrelife'].diff()`. Equals 1 on a normal lap, big negative jump on the lap after a pit. Captures "did they just pit?" cleanly.
- [ ] `laps_since_last_pit` — derivable from `tyrelife`, but an explicit cumcount-since-pit-event sometimes splits cleaner. Try if `tyrelife_diff` shows up in importance.
- [ ] `laptime_prev` and `laptime_diff` — previous lap's lap time and the difference vs. current. Slowing down before a pit is a known pattern.
- [ ] `laptime_3lap_slope` — slope of last 3 laps' lap times via `groupby([...])['laptime_(s)'].rolling(3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])`. Captures degradation trend.
- [ ] `position_change_3lap` — sum of recent position changes. Drivers losing positions may pit to undercut.
- [ ] `cumulative_degradation_diff` — lap-over-lap change (the rate, not the level).

#### Combination Aggregates (Lower Priority)
Aggregations across combinations the tree can't isolate cheaply, even with native categoricals:
- [ ] `(driver, race)` mean/median pit lap from training data — captures driver-track preferences.
- [ ] `(compound, race)` typical first-pit lap — pit window per compound per track.
- [ ] **Target encoding for `driver`** with proper CV fold separation. With hundreds of driver codes, native categorical handling may underperform a smoothed target encoding. Use out-of-fold encoding to avoid leakage.

#### What NOT to Try (already-known dead ends from S6E5)
- ❌ `wet_race` (Exp 1) — redundant with `compound` (INTER/WET values already encode this).
- ❌ `avg_stint_per_race` (Exp 1) — single scalar per race, redundant with native categorical handling of `race`.
- **General rule from Exp 1:** aggregating a numeric by a categorical *already in the model* is usually redundant for tree models. Save aggregations for combinations of categoricals or for time/sequence info.

### Encoding Strategies
- [ ] XGBoost native categorical support (`enable_categorical=True`) — usually best for tree models
- [ ] OHE only for correlation analysis or non-tree models

### Model Options
- [ ] XGBoost (default starting point)
- [ ] LightGBM
- [ ] CatBoost
- [ ] Hyperparameter tuning with Optuna (TPE sampler + median pruner)
- [ ] Ensemble/stacking

### Advanced
- [ ] Blend in original dataset if available
- [ ] Pseudo-labeling with confident predictions
- [ ] Feature selection (drop low-importance features)

---

## Lessons Carried Over From Prior Competitions

### Process / Workflow
- **Match training objective to the competition metric.** Single biggest lever in S6E4 was using balanced sample weights when the metric was balanced accuracy (+0.007 CV).
- **Trust your CV when std is tight.** With 600K+ rows, 10-fold CV gives stable enough scores to compare experiments. Track both CV and LB and confirm they move together.
- **CV up but LB down = overfitting warning.** When this happens, suspect features that fit the training distribution too tightly (e.g., hand-crafted rules from a smaller original dataset that don't transfer to synthetic data).
- **Always train final model on full training set.** Hold out only for validation/tuning. The full-data model usually performs slightly better on LB.

### Feature Engineering for Tree Models
- **Trees find rectangular splits automatically.** Pre-engineered booleans (e.g., `feature > threshold`) add nothing — XGBoost will find better thresholds itself.
- **Pre-computed products like Temperature × Humidity also rarely help.** Trees can approximate these with combinations of splits.
- **Do help:** ratios, group-based aggregations, target encoding. These encode info trees *can't* discover from rectangular splits alone.

### Hyperparameter Tuning
- **Optuna with TPE sampler + median pruner is efficient.** 30 trials with 5-fold CV during tuning, then validate winner on 10-fold.
- **Use early stopping per fold** to let each fold pick its own `n_estimators`. For final fit on all data (no validation), either remove early stopping (default 100 trees) or set `n_estimators` to roughly the average best_iteration seen during CV.
- **Pass `sample_weight` per fold** (`compute_sample_weight('balanced', y_train)`) — not globally — and pass `sample_weight_eval_set` so early stopping evaluates on weighted validation.

### XGBoost Specifics (3.x)
- `enable_categorical=True` works well; convert categorical columns with `.astype('category')`.
- `early_stopping_rounds` goes on the constructor, not in `.fit()`.
- Default `tree_method='hist'` is fast on CPU. For GPU, add `'device': 'cuda'`.

### Pitfalls Hit Before
- **Reassigning `df` between train and test prep** breaks final model cell — use captured `X, y` variables instead of `df[features], df[target]`.
- **Submission save guard:** check that new submission's value_counts differ from last submission before saving — avoids accidental duplicate submissions. *For float predictions (AUC), use `np.allclose` instead — `value_counts` equality is meaningless on continuous outputs.*

### ROC-AUC Specifics (S6E5)
- **Submit probabilities, not 0/1.** `predict_proba(X)[:, 1]`, never `predict()`. Hard labels collapse the ROC into a step function and tank the score.
- **Skip `compute_sample_weight('balanced', ...)`.** AUC is rank-based and threshold-independent — balanced weights distort the loss surface without helping the metric. Same for `scale_pos_weight`. Save these for metrics that *do* respond to class weighting (balanced accuracy, F1).
- **Skip threshold tuning, skip calibration.** AUC only cares about rank order — a monotonic transform of probabilities (sigmoid, isotonic, etc.) leaves AUC unchanged.
- **Use `eval_metric='auc'`** so early stopping picks the best AUC iteration, not the best logloss iteration.
- **Class imbalance still hurts learning.** AUC won't penalize a model that ignores the minority class on a balanced metric, but the model can still struggle to find minority-class signal. If CV AUC plateaus, *then* consider whether the model is even learning the positives — not whether to reweight the metric.
- **Group-aware CV matters here.** Random folds on per-(driver, race, lap) data leak race context — model learns "this race pits around lap N" instead of generalizable patterns. Use `StratifiedGroupKFold` with groups = `race + '_' + year`.

---

## Experiment Log

| # | Description | CV (AUC) | LB (AUC) | Notes |
|---|------------|----------|----------|-------|
| 0 | Baseline: XGBClassifier defaults, StratifiedGroupKFold(10) by (race, year), early stopping=50, no FE beyond pipeline | 0.92156 ± 0.00995 | 0.93671 | LB > CV by +0.015 — see "CV-LB Gap" below |
| 1 | + `wet_race`, `avg_stint_per_race` features | 0.92137 ± 0.00979 | 0.93649 | Δ within std (noise). Features likely redundant with `race` + `compound` — tree already learns these via native categorical splits |
| 2 | + concat original dataset (`f1_strategy_dataset_v4.csv`), drop `normalized_tyrelife` (not in train/test) | 0.91452 ± 0.01443 | 0.93844 | **CV-LB divergence**: CV down −0.0070, LB up +0.0017. CV std also widened (0.010 → 0.014). See "CV-LB Divergence (Exp 2)" below. |
| 3 | Exp 2 + `sample_weight=2.0` on original rows, `1.0` on train rows | 0.91276 ± 0.01627 | 0.93851 | Weights confirmed applied (per-fold sum ~577k). CV slightly down vs Exp 2 (−0.0018), std widened (0.0144 → 0.0163). LB +0.00007 vs Exp 2 — essentially identical. 2x weight isn't moving the needle. |
| 4 | Exp 3 + final fit uses `n_estimators = np.max(best_iters) = 100` instead of `np.mean = 38` | 0.91276 ± 0.01627 (CV unchanged) | 0.93873 | LB +0.00022 vs Exp 3. Confirms hypothesis: final model was undertrained at mean(best_iters). Cheap win. |
| 5 | Exp 4 + bump `n_estimators` to `int(np.max(best_iters) * 10/9) = 111` | (CV unchanged) | 0.93864 | LB −0.00009 vs Exp 4. Within noise. n_estimators lever is exhausted at ~100 trees. |
| 6 | Exp 4 − drop `driver` from features | 0.92043 ± 0.01481 | 0.94731 | **Massive: CV +0.0077, LB +0.0086.** Both moved together. Dropping the noisy high-cardinality feature recovered real generalization. |
| 7 | Exp 6 − drop `wet_race` and `avg_stint_per_race` | 0.92140 ± 0.01483 | 0.94723 | CV +0.001 (noise-ish), LB −0.00008 (essentially flat). Custom features were marginally noisy, no real loss from dropping. Cleaner feature set; keep dropped. |
| 8 | Exp 7 + frequency encoding on `year`, `compound`, `race` | 0.92299 ± 0.01376 | 0.94100 | **CV/LB divergence (overfit signal).** CV +0.0016, LB −0.0062. Importance ranking misleading: year_freq #2, compound_freq mid, race_freq last; combined-with-original showed mild uplift, but model didn't generalize. Rollback to Exp 7's feature set. |
| 9 | Exp 7 + treat `year` as categorical (add to `categorical_features`) | 0.92414 ± 0.01550 | 0.94778 | **Both directions: CV +0.0027, LB +0.0055.** Categorical partition splits isolate 2023 in one node vs the two splits numeric year needs. Cleaner version of what year_freq was attempting, without the overfit. |
