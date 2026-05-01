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
- (User to fill in after EDA)

## CV-LB Gap Analysis (after Exp 0)
- **CV: 0.92156, LB: 0.93671 — LB is +0.015 higher than CV.** Inverse of the usual overfit pattern.
- **Likely cause:** `StratifiedGroupKFold` by `(race, year)` holds *entire races* out of training. The model never sees that race's specific context (track-specific tyre wear, typical pit windows, traffic patterns) during validation, so CV measures generalization to *unseen races*. The Kaggle test set was almost certainly split row-wise from the same race-year pool as train, so on the LB the model sees familiar races. CV is harder than the LB scenario — exactly what we want.
- **Implication:** trust CV as a *lower bound*. Improvements on CV will translate to LB; a CV bump is real signal, not just LB noise.
- **Sanity check worth doing once:** `set(test_df.Race + test_df.Year) - set(train_df.Race + train_df.Year)` — if it's empty, test races are all seen in train (random split, validates the theory above). If non-empty, group CV is even more important and the gap may shrink.
- **Don't switch back to plain `StratifiedKFold`** chasing a higher CV — it would inflate CV without helping LB, and you'd lose the early-warning system for race-context overfitting.

## Open Questions (flagged for EDA)
- **CV strategy**: data is per-(Driver, Race, Year, Lap). Random KFold will leak race/driver context across folds. Likely want **GroupKFold by `Race`** or by `(Race, Year)`. Worth checking how train/test were split — if test contains races not in train, group CV is mandatory.
- **Driver coding**: train sample shows both `D109` (anonymized) and `VER` (real abbreviation). Are these from different sources, or is the dataset synthetic-with-some-real-codes? Affects whether `Driver` is a usable feature or just noise.
- **Class balance**: pit events are rare per lap — likely 5–10% positive, but confirm.

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
