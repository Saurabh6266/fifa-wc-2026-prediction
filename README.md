# 🏆 FIFA World Cup 2026 — ML Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.ai)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-ff4b4b)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Model accuracy: 56.5% outcome accuracy on 69 real 2026 WC group-stage matches** — vs 43.5% naive FIFA-rank baseline

A rigorous, production-quality machine learning pipeline to predict 2026 FIFA World Cup match outcomes, scorelines, corners, yellow cards, red cards, and full tournament bracket progression across all 104 matches.

---

## 🏅 Key Predictions (from 50,000 Monte Carlo simulations)

| | Team | Probability |
|---|---|---|
| 🥇 Tournament Winner | **Argentina** | 27.0% |
| 🥈 Runner-up | **Japan** | — |
| 🥉 Semi-finalists | Argentina, Japan, Spain, Portugal | — |

**Predicted Final:** Argentina vs Japan · Predicted winner: **Argentina** (71.4%)

---

## 📁 Project Structure

```
fifa-wc-2026-prediction/
├── app.py                     ← Gradio web app (for deployment)
├── requirements.txt           ← Full pipeline dependencies
├── requirements_app.txt       ← Deployment-only dependencies
├── data/
│   ├── raw/                   ← Input CSVs (gitignored)
│   └── processed/             ← Generated features & model outputs
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_goals_outcomes.ipynb
│   ├── 05_model_corners_cards.ipynb
│   ├── 06_tournament_simulation.ipynb
│   └── 07_predictions_and_validation.ipynb
├── outputs/
│   └── predictions/
│       ├── SUMMARY.md                    ← Full prediction report
│       ├── group_stage_predictions.csv   ← All 72 group matches
│       ├── knockout_predictions.csv      ← R32 → Final bracket
│       └── corners_cards_predictions.csv ← Corners & cards per match
└── src/
    ├── data_loader.py          ← Data ingestion & normalization
    ├── dixon_coles.py          ← Dixon-Coles bivariate Poisson model
    ├── elo_calculator.py       ← Elo rating system (1872–2026)
    ├── feature_engineer.py     ← Rolling stats, Elo features, H2H
    ├── models.py               ← XGBoost classifier pipeline
    ├── tournament_simulator.py ← Monte Carlo bracket simulator
    ├── evaluator.py            ← Backtesting & metrics
    └── name_normalizer.py      ← Team name harmonisation across datasets
```

---

## 🧠 Methodology

### Models

| Model | Purpose | Key Parameters |
|---|---|---|
| **Dixon-Coles** | Scoreline probabilities | xi=0.003, fitted on 2015–2026 |
| **Elo Ratings** | Team strength ranking | K=60 (WC), K=40 (Confed), K=20 (Friendly) |
| **XGBoost** | Outcome probabilities | 25+ engineered features |
| **Monte Carlo** | Bracket simulation | N=50,000 iterations |

### Pipeline Overview
```
Raw Data → Elo Ratings → Feature Engineering → XGBoost (outcomes)
                      ↓
               Dixon-Coles Fit → Monte Carlo Simulation → Bracket Predictions
```

### Validation Results (72 Real 2026 WC Group Stage Matches)

| Metric | Our Model | Naive Baseline |
|---|---|---|
| Outcome Accuracy | **56.5%** | 43.5% |
| Goal-Diff Accuracy | 13.0% | 10.1% |
| Exact Score Accuracy | 4.3% | 5.8% |
| Matches Evaluated | 69 of 72 | — |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Saurabh6266/fifa-wc-2026-prediction.git
cd fifa-wc-2026-prediction
pip install -r requirements.txt
```

### 2. Add Data Files
Place the following CSVs in `data/raw/`:
- `results.csv` — International football results 1872–2026 (Kaggle)
- `FIFA_Rankings.csv`, `schedule_2026.csv`, `shootouts.csv`, `goalscorers.csv`

### 3. Run the Pipeline (in order)
```bash
jupyter notebook notebooks/
# Run notebooks 01 → 07 in sequence
```

### 4. Launch the Web App Locally
```bash
pip install gradio
python app.py
```

---

## 🌐 Deploy to Hugging Face Spaces (Recommended — Free)

The app only reads pre-computed CSVs — no GPU/training needed at runtime.

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Upload this entire repo (or connect your GitHub)
4. Set `requirements_app.txt` as the requirements file in Space settings
5. The app will auto-deploy and give you a public URL!

---

## 📊 Data Sources

| Dataset | Source |
|---|---|
| International results 1872–2026 | [Kaggle — martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| FIFA Rankings & WC schedule | [Kaggle — Club Football dataset](https://www.kaggle.com/) |
| Penalty shootout data | Included in results dataset |

---

## 🔮 Future Work / Validation (Post-Tournament)

To validate knockout predictions against real results as the tournament progresses:
1. Download live results from [football-data.org](https://www.football-data.org/) or [FIFA official](https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026)
2. Create `data/raw/wc2026_actuals.csv` with columns: `match_id, home_team, away_team, home_score, away_score, date`
3. Run Notebook 07 which auto-detects and backtests against real results

---

## 👤 Author

**Saurabh Gupta**  
Built as a personal ML learning project during the live 2026 FIFA World Cup

---

*Pipeline: Elo + Dixon-Coles ensemble calibrated on 2022 WC holdout. Corners & cards: Poisson regression.*