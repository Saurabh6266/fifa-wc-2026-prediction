# 🏆 FIFA World Cup 2026 — ML Prediction Pipeline

A rigorous, production-quality machine learning pipeline to predict 2026 FIFA World Cup
match outcomes, scorelines, corners, yellow cards, red cards, and full tournament
bracket progression across all 104 matches.

---

## 📁 Project Structure

```
fifa-wc-2026-prediction/
├── data/
│   ├── raw/               ← all input CSVs
│   └── processed/         ← generated feature matrices & results
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_goals_outcomes.ipynb
│   ├── 05_model_corners_cards.ipynb
│   ├── 06_tournament_simulation.ipynb
│   └── 07_predictions_and_validation.ipynb
├── src/
│   ├── __init__.py
│   ├── name_normalizer.py     ← team name standardization
│   ├── data_loader.py         ← data loading & team registry
│   ├── elo_calculator.py      ← national team Elo ratings
│   ├── dixon_coles.py         ← bivariate Poisson score model
│   ├── feature_engineer.py    ← match feature matrix builder
│   ├── models.py              ← XGBoost + ensemble + corners/cards
│   ├── tournament_simulator.py ← Monte Carlo bracket simulation
│   └── evaluator.py           ← backtest & report generator
├── outputs/
│   ├── predictions/           ← final prediction CSVs + SUMMARY.md
│   ├── plots/                 ← all EDA and validation plots
│   └── model_artifacts/       ← saved model files
└── requirements.txt
```

---

## 🚀 Quick Start (Local Setup)

```bash
git clone https://github.com/Saurabh6266/fifa-wc-2026-prediction.git

cd fifa-wc-2026-prediction

# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter Notebook
jupyter notebook notebooks/
```

Run notebooks **in order** (01 → 07). Each builds on the previous and saves outputs to `data/processed/` and `outputs/`.

---

## 🧠 Pipeline Architecture

### Phase 1 — Data Loading (Notebook 01)
- Normalizes 50+ team name variants across 7 data sources
- Validates results.csv (47K+ matches, 1872–2026)
- Builds master team registry with FIFA rankings

### Phase 2 — EDA (Notebook 02)
- Goals distributions vs Poisson approximation
- Home advantage at neutral vs non-neutral venues
- Confederation strength heatmap
- WC-specific patterns (group stage vs knockout)
- Team profiles for all 48 WC 2026 teams

### Phase 3 — Feature Engineering (Notebook 03)
- **Elo ratings** built from scratch (all results since 1872, K-factor by tournament)
- No data leakage: all features use only pre-match information
- Rolling form scores, H2H records, goals averages

### Phase 4 — Goals & Outcome Models (Notebook 04)
- **Dixon-Coles bivariate Poisson** with time-decay weighting (xi tuned on 2022 WC)
- **XGBoost classifier** with Optuna hyperparameter tuning (50 trials, 5-fold CV)
- **Ensemble**: 60% Dixon-Coles + 40% XGBoost, calibrated via isotonic regression
- Validated on 2022 FIFA World Cup (RPS, Brier score, accuracy)

### Phase 5 — Corners & Cards (Notebook 05)
- Corners: prior-based (9.8/match) + attacking style adjustment, clipped [6,14]
- Yellow cards: Poisson regression with confederation pair + knockout adjustment
- Red cards: **Always 0** (modal value, minimises prediction error)

### Phase 6 — Tournament Simulation (Notebook 06)
- Monte Carlo N=50,000 iterations
- Group standings: pts → GD → GF → discipline → random tiebreaker
- Best 3rd selection: 8 of 12 advance by ranking
- Knockout: Dixon-Coles sampling, ties resolved by penalty shootout
- Outputs: win probabilities for all 104 matches

### Phase 7 — Predictions & Validation (Notebook 07)
- Final prediction tables (group stage + knockout)
- Backtest against already-played WC 2026 matches
- Comparison vs naive baseline
- SUMMARY.md with key predictions and upset alerts

---

## 📊 Key Data Sources

| File | Source | Use |
|---|---|---|
| `results.csv` | martj42 (Kaggle) | Primary training data |
| `FIFA_Rankings.csv` | Manual + corrected with official 2026 rankings | FIFA points & confederation |
| `group_fixtures.csv` | Derived from schedule_2026.csv + FIFA draw | **Ground truth** for groups |
| `knockout_slots.csv` | Built from 2026 WC bracket | Bracket structure |
| `WC_historical_stats.csv` | piterfm (Kaggle) | WC match details 1930-2022 |
| `fifa_ranking_2026-06-08.csv` | piterfm (Kaggle) | Official June 2026 FIFA rankings |
| `shootouts.csv` | martj42 (Kaggle) | Penalty shootout calibration |

---

## ⚠️ Important Notes

1. **EloRatings.csv** (club Elo) must NOT be used for national team ratings.
2. **Group column in FIFA_Rankings.csv** is ignored — groups from `group_fixtures.csv` only.
3. **Playoff slots** (6 undetermined qualifiers) use median Elo of likely candidates.
4. Tournament is **live** as of June 28, 2026 — predictions can be backtested against real results.

---

## 🔢 Expected Validation Performance

| Metric | Target | Notes |
|---|---|---|
| Outcome Accuracy | ~50–55% | Realistic for football; >60% = likely data leakage |
| RPS (Dixon-Coles) | <0.20 | Lower is better |
| Exact Score Accuracy | ~8–12% | Hard task; random baseline ~5% |

---

*Random seed: 42 everywhere for full reproducibility.*