"""
src/evaluator.py

Phase 7: Prediction validation and backtest engine.

Handles:
- Backtest against already-played group stage results
- Outcome accuracy, goal difference accuracy, exact score accuracy
- Corners and yellow cards MAE
- Comparison vs naive baseline
- Summary Markdown report generation
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

OUTPUTS_PREDICTIONS = Path(__file__).parent.parent / "outputs" / "predictions"
OUTPUTS_PREDICTIONS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------

class Backtest:
    """
    Compares model predictions against actual results for played matches.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Model predictions with columns:
        match_id, home_team, away_team,
        predicted_home_goals, predicted_away_goals,
        predicted_outcome ('home'|'draw'|'away'),
        corners, yellow_cards, red_cards.
    actuals_df : pd.DataFrame
        Actual results. Can be results.csv filtered to WC 2026 dates,
        or group_fixtures with actual scores filled in.
        Required columns: home_team, away_team, home_score, away_score.
    """

    def __init__(self, predictions_df: pd.DataFrame, actuals_df: pd.DataFrame):
        self.preds = predictions_df.copy()
        self.actuals = actuals_df.copy()
        self._merged = None

    def merge(self) -> pd.DataFrame:
        """
        Inner-join predictions with actuals on home_team + away_team.

        Returns
        -------
        pd.DataFrame  Merged dataframe for metric computation.
        """
        actual_cols = ["home_team", "away_team", "home_score", "away_score"]
        available = [c for c in actual_cols if c in self.actuals.columns]

        merged = self.preds.merge(
            self.actuals[available],
            on=["home_team", "away_team"],
            how="inner"
        )

        if len(merged) == 0:
            logger.warning("No matches found in backtest merge. Check team name normalization.")
            return merged

        # Derive actual outcome
        merged["actual_outcome"] = np.where(
            merged["home_score"] > merged["away_score"], "home",
            np.where(merged["home_score"] == merged["away_score"], "draw", "away")
        )
        merged["actual_goal_diff"] = merged["home_score"] - merged["away_score"]
        merged["pred_goal_diff"]   = merged["predicted_home_goals"] - merged["predicted_away_goals"]
        merged["exact_score_correct"] = (
            (merged["predicted_home_goals"] == merged["home_score"]) &
            (merged["predicted_away_goals"] == merged["away_score"])
        ).astype(int)

        self._merged = merged
        logger.info(f"Backtest: {len(merged)} matched predictions")
        return merged

    def compute_metrics(self) -> dict:
        """
        Compute all validation metrics.

        Returns
        -------
        dict
            outcome_accuracy, goal_diff_accuracy, exact_score_accuracy,
            corners_mae, yellow_cards_mae, n_matches.
        """
        if self._merged is None:
            self.merge()
        m = self._merged

        if len(m) == 0:
            return {"n_matches": 0, "note": "No matched predictions"}

        n = len(m)

        # Outcome accuracy
        outcome_acc = (m["predicted_outcome"] == m["actual_outcome"]).sum() / n

        # Goal difference accuracy (exact)
        gd_acc = (m["pred_goal_diff"] == m["actual_goal_diff"]).sum() / n

        # Exact score accuracy
        exact_acc = m["exact_score_correct"].sum() / n

        # Corners MAE
        corners_mae = None
        if "corners" in m.columns and "actual_corners" in m.columns:
            corners_mae = np.abs(m["corners"] - m["actual_corners"]).mean()

        # Yellow cards MAE
        yellow_mae = None
        if "yellow_cards" in m.columns and "actual_yellow_cards" in m.columns:
            yellow_mae = np.abs(m["yellow_cards"] - m["actual_yellow_cards"]).mean()

        return {
            "n_matches":            n,
            "outcome_accuracy":     round(float(outcome_acc), 4),
            "goal_diff_accuracy":   round(float(gd_acc), 4),
            "exact_score_accuracy": round(float(exact_acc), 4),
            "corners_mae":          round(float(corners_mae), 3) if corners_mae is not None else "N/A",
            "yellow_cards_mae":     round(float(yellow_mae), 3) if yellow_mae is not None else "N/A",
        }

    def compute_naive_metrics(self) -> dict:
        """
        Compute metrics for the naive baseline:
        - Higher-ranked (lower FIFA rank number) team always wins
        - Scoreline: 1-0
        - Corners: 9, Yellow cards: 3

        Returns
        -------
        dict  Same structure as compute_metrics().
        """
        if self._merged is None:
            self.merge()
        m = self._merged

        if len(m) == 0:
            return {}

        n = len(m)

        # Naive prediction: home wins if home_rank <= away_rank
        naive_outcome = np.where(
            m.get("fifa_rank_home", np.nan) <= m.get("fifa_rank_away", np.nan),
            "home", "away"
        )
        naive_outcome_acc = (naive_outcome == m["actual_outcome"]).sum() / n
        naive_exact = ((m["home_score"] == 1) & (m["away_score"] == 0)).sum() / n

        return {
            "n_matches":            n,
            "outcome_accuracy":     round(float(naive_outcome_acc), 4),
            "goal_diff_accuracy":   round(float(((m["home_score"] - m["away_score"]) == 1).sum() / n), 4),
            "exact_score_accuracy": round(float(naive_exact), 4),
            "corners_mae":          "N/A (baseline=9)",
            "yellow_cards_mae":     "N/A (baseline=3)",
        }

    def report(self) -> str:
        """
        Generate a human-readable backtest report string.
        """
        model_metrics = self.compute_metrics()
        naive_metrics = self.compute_naive_metrics()

        lines = [
            "=" * 55,
            "BACKTEST REPORT — WC 2026 GROUP STAGE",
            "=" * 55,
            f"Matched predictions: {model_metrics.get('n_matches', 0)} matches",
            "",
            "MODEL PERFORMANCE:",
            f"  Outcome accuracy:      {model_metrics.get('outcome_accuracy', 'N/A'):.1%}",
            f"  Goal-diff accuracy:    {model_metrics.get('goal_diff_accuracy', 'N/A'):.1%}",
            f"  Exact score accuracy:  {model_metrics.get('exact_score_accuracy', 'N/A'):.1%}",
            f"  Corners MAE:           {model_metrics.get('corners_mae', 'N/A')}",
            f"  Yellow cards MAE:      {model_metrics.get('yellow_cards_mae', 'N/A')}",
            "",
            "NAIVE BASELINE:",
            f"  Outcome accuracy:      {naive_metrics.get('outcome_accuracy', 'N/A'):.1%}",
            f"  Goal-diff accuracy:    {naive_metrics.get('goal_diff_accuracy', 'N/A'):.1%}",
            f"  Exact score accuracy:  {naive_metrics.get('exact_score_accuracy', 'N/A'):.1%}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PREDICTION ASSEMBLER
# ---------------------------------------------------------------------------

class PredictionAssembler:
    """
    Assembles the final prediction tables for group stage and knockout rounds.
    """

    def __init__(self, dc_model, xgb_model, ensemble, corners_model, yellows_model):
        self.dc = dc_model
        self.xgb = xgb_model
        self.ensemble = ensemble
        self.corners = corners_model
        self.yellows = yellows_model

    def assemble_group_predictions(self, wc_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the complete group stage prediction table.

        Parameters
        ----------
        wc_features_df : pd.DataFrame
            WC 2026 feature matrix (from feature_engineer.py).

        Returns
        -------
        pd.DataFrame
            72 rows × prediction columns.
        """
        rows = []
        for _, feat in wc_features_df.iterrows():
            home = feat["home_team"]
            away = feat["away_team"]
            is_ph = bool(feat.get("is_placeholder", 0))

            # Dixon-Coles
            dc_probs = self.dc.predict_outcome_probs(home, away, neutral=True)
            dc_h, dc_a = self.dc.predict_scoreline(home, away, neutral=True)

            # XGBoost (skip for placeholders)
            if not is_ph and self.xgb.model is not None:
                try:
                    xgb_probs = self.xgb.predict_outcome_probs(feat)
                except Exception:
                    xgb_probs = {"home": 1/3, "draw": 1/3, "away": 1/3}
            else:
                xgb_probs = {"home": 1/3, "draw": 1/3, "away": 1/3}

            # Ensemble
            ens = self.ensemble.predict(dc_probs, xgb_probs)
            predicted_outcome = max(ens, key=ens.get)

            # Consistency check: if goals imply different winner, use goals
            if dc_h > dc_a:
                goals_outcome = "home"
            elif dc_h < dc_a:
                goals_outcome = "away"
            else:
                goals_outcome = "draw"

            inconsistent = (goals_outcome != predicted_outcome)

            # Secondary predictions
            corners = self.corners.predict_from_row(feat)
            yellows = self.yellows.predict_from_row(feat)

            rows.append({
                "match_id":             feat.get("match_id", ""),
                "group":                feat.get("group", ""),
                "date_utc":             feat.get("date_utc", ""),
                "venue":                feat.get("venue", ""),
                "home_team":            home,
                "away_team":            away,
                "is_placeholder":       int(is_ph),
                "predicted_home_goals": dc_h,
                "predicted_away_goals": dc_a,
                "predicted_outcome":    predicted_outcome,
                "corners":              corners,
                "yellow_cards":         yellows,
                "red_cards":            0,
                "p_home_win":           round(ens["home"], 4),
                "p_draw":               round(ens["draw"], 4),
                "p_away_win":           round(ens["away"], 4),
                "dc_home_win":          round(dc_probs["home"], 4),
                "dc_draw":              round(dc_probs["draw"], 4),
                "dc_away_win":          round(dc_probs["away"], 4),
                "exp_home_goals":       round(dc_probs.get("exp_home_goals", dc_h), 3),
                "exp_away_goals":       round(dc_probs.get("exp_away_goals", dc_a), 3),
                "outcome_inconsistency": int(inconsistent),
            })

        df = pd.DataFrame(rows)
        out = OUTPUTS_PREDICTIONS / "group_stage_predictions.csv"
        df.to_csv(out, index=False)
        logger.info(f"Saved group_stage_predictions.csv → {out}")
        return df

    def assemble_knockout_predictions(self, ko_sim_results: dict,
                                       wc_features_df: pd.DataFrame,
                                       knockout_slots: pd.DataFrame) -> pd.DataFrame:
        """
        Build knockout stage prediction table from simulation results.

        Parameters
        ----------
        ko_sim_results : dict  Output from TournamentSimulator.run().
        wc_features_df : pd.DataFrame  For corners/cards features of WC teams.
        knockout_slots : pd.DataFrame

        Returns
        -------
        pd.DataFrame  32 rows (match IDs 73-104).
        """
        ko_preds = ko_sim_results.get("ko_predictions", pd.DataFrame())
        elo_dict = {}  # could inject elo_ratings here if needed

        # Add corners and yellow cards to knockout predictions
        rows = []
        for _, row in ko_preds.iterrows():
            home = row.get("predicted_home", "TBD")
            away = row.get("predicted_away", "TBD")

            # Look up features for these teams
            h_feat = wc_features_df[wc_features_df["home_team"] == home]
            a_feat = wc_features_df[wc_features_df["away_team"] == away]

            h_gf = h_feat["goals_scored_avg_home"].mean() if len(h_feat) else 1.2
            h_gc = h_feat["goals_conceded_avg_home"].mean() if len(h_feat) else 1.2
            a_gf = a_feat["goals_scored_avg_away"].mean() if len(a_feat) else 1.2
            a_gc = a_feat["goals_conceded_avg_away"].mean() if len(a_feat) else 1.2
            elo_diff = 0.0

            corners = self.corners.predict(h_gf, a_gf, h_gc, a_gc, elo_diff)
            yellows = self.yellows.predict(
                conf_home="UEFA", conf_away="UEFA",
                is_knockout=True, elo_diff=elo_diff
            )

            # Scoreline prediction for the modal matchup
            try:
                dc_h, dc_a = self.dc.predict_scoreline(home, away, neutral=True)
                dc_probs = self.dc.predict_outcome_probs(home, away, neutral=True)
                winner_goals = home if dc_h > dc_a else (away if dc_a > dc_h else home)
            except Exception:
                dc_h, dc_a = 1, 0
                winner_goals = home

            rows.append({
                "match_id":             row.get("match_id", ""),
                "round":                row.get("round", ""),
                "date_utc":             row.get("date_utc", ""),
                "venue":                row.get("venue", ""),
                "slot_home":            row.get("slot_home", ""),
                "slot_away":            row.get("slot_away", ""),
                "predicted_home":       home,
                "predicted_away":       away,
                "predicted_home_goals": dc_h,
                "predicted_away_goals": dc_a,
                "match_winner":         row.get("predicted_winner", winner_goals),
                "home_win_prob":        row.get("home_win_prob", 0.5),
                "away_win_prob":        row.get("away_win_prob", 0.5),
                "corners":              corners,
                "yellow_cards":         yellows,
                "red_cards":            0,
            })

        df = pd.DataFrame(rows)
        out = OUTPUTS_PREDICTIONS / "knockout_predictions.csv"
        df.to_csv(out, index=False)
        logger.info(f"Saved knockout_predictions.csv → {out}")
        return df


# ---------------------------------------------------------------------------
# SUMMARY REPORT GENERATOR
# ---------------------------------------------------------------------------

def generate_summary_report(winner_probs: pd.DataFrame,
                              ko_preds: pd.DataFrame,
                              group_preds: pd.DataFrame,
                              group_stage_probs: pd.DataFrame,
                              model_metrics: dict,
                              naive_metrics: dict,
                              output_path: str = None) -> str:
    """
    Generate the SUMMARY.md file with tournament predictions.

    Parameters
    ----------
    winner_probs : pd.DataFrame  Tournament winner probabilities.
    ko_preds : pd.DataFrame      Knockout predictions.
    group_preds : pd.DataFrame   Group stage predictions.
    group_stage_probs : pd.DataFrame  Group qualification probabilities.
    model_metrics : dict  Backtest metrics.
    naive_metrics : dict  Baseline metrics.
    output_path : str  Defaults to outputs/predictions/SUMMARY.md.

    Returns
    -------
    str  Markdown content.
    """
    if output_path is None:
        output_path = OUTPUTS_PREDICTIONS / "SUMMARY.md"

    # Final match
    final_row = ko_preds[ko_preds["round"] == "Final"]
    semis_rows = ko_preds[ko_preds["round"] == "Semi-final"]
    quarters_rows = ko_preds[ko_preds["round"] == "Quarter-final"]

    champ = winner_probs.iloc[0] if len(winner_probs) else None
    finalist1 = final_row.iloc[0]["predicted_home"] if len(final_row) else "TBD"
    finalist2 = final_row.iloc[0]["predicted_away"] if len(final_row) else "TBD"
    final_score_h = final_row.iloc[0].get("predicted_home_goals", "?") if len(final_row) else "?"
    final_score_a = final_row.iloc[0].get("predicted_away_goals", "?") if len(final_row) else "?"

    # Semifinalists
    semifinalists = []
    for _, r in semis_rows.iterrows():
        semifinalists.extend([r.get("predicted_home", "TBD"), r.get("predicted_away", "TBD")])

    # Quarterfinalists
    quarterfinalists = []
    for _, r in quarters_rows.iterrows():
        quarterfinalists.extend([r.get("predicted_home", "TBD"), r.get("predicted_away", "TBD")])

    # Upset alerts: lower-ranked team has >45% win probability
    upsets = []
    for _, row in group_preds.iterrows():
        rank_h = row.get("fifa_rank_home", 999) if "fifa_rank_home" in row else 999
        rank_a = row.get("fifa_rank_away", 999) if "fifa_rank_away" in row else 999
        p_upset = row.get("p_away_win", 0) if rank_h < rank_a else row.get("p_home_win", 0)
        upset_team = row["away_team"] if rank_h < rank_a else row["home_team"]
        fav_team   = row["home_team"] if rank_h < rank_a else row["away_team"]
        if p_upset > 0.45 and upset_team != fav_team:
            upsets.append({
                "match": f"{row['home_team']} vs {row['away_team']}",
                "underdog": upset_team,
                "favourite": fav_team,
                "upset_prob": p_upset,
            })
    upsets = sorted(upsets, key=lambda x: -x["upset_prob"])[:5]

    # Build markdown
    lines = [
        "# 🏆 FIFA World Cup 2026 — Prediction Summary",
        "",
        f"> Generated by the WC2026 ML Prediction Pipeline | Date: 2026-06-28",
        "",
        "---",
        "",
        "## 🥇 Predicted Tournament Winner",
        "",
    ]

    if champ is not None:
        lines += [
            f"**{champ['team']}** — Win probability: **{champ['win_probability']:.1%}**",
            "",
            f"- Finalist probability:    {champ.get('finalist_prob', '?'):.1%}",
            f"- Semifinalist probability: {champ.get('semi_prob', '?'):.1%}",
            "",
        ]

    lines += [
        "## 🎯 Predicted Final",
        "",
        f"**{finalist1}** vs **{finalist2}** — Predicted score: {final_score_h}–{final_score_a}",
        "",
        "## 🏅 Predicted Semifinalists",
        "",
    ]
    for t in sorted(set(semifinalists)):
        lines.append(f"- {t}")
    lines += [""]

    lines += [
        "## 🎖️ Predicted Quarterfinalists",
        "",
    ]
    for t in sorted(set(quarterfinalists)):
        lines.append(f"- {t}")
    lines += [""]

    lines += [
        "## ⚡ Top 5 Upset Alerts",
        "*(matches where the lower-ranked team has >45% win probability)*",
        "",
    ]
    if upsets:
        for u in upsets:
            lines.append(f"- **{u['match']}**: {u['underdog']} ({u['upset_prob']:.0%} to upset {u['favourite']})")
    else:
        lines.append("- No major upsets predicted at >45% probability")
    lines += [""]

    lines += [
        "## 📊 Model Validation Metrics (Group Stage Backtest)",
        "",
        "| Metric | Our Model | Naive Baseline |",
        "|---|---|---|",
        f"| Outcome Accuracy | {model_metrics.get('outcome_accuracy', 'N/A'):.1%} | {naive_metrics.get('outcome_accuracy', 'N/A'):.1%} |",
        f"| Goal-Diff Accuracy | {model_metrics.get('goal_diff_accuracy', 'N/A'):.1%} | {naive_metrics.get('goal_diff_accuracy', 'N/A'):.1%} |",
        f"| Exact Score Accuracy | {model_metrics.get('exact_score_accuracy', 'N/A'):.1%} | {naive_metrics.get('exact_score_accuracy', 'N/A'):.1%} |",
        f"| Corners MAE | {model_metrics.get('corners_mae', 'N/A')} | 9 (fixed) |",
        f"| Yellow Cards MAE | {model_metrics.get('yellow_cards_mae', 'N/A')} | 3 (fixed) |",
        f"| Matches Evaluated | {model_metrics.get('n_matches', 0)} | — |",
        "",
        "## 🔧 Key Assumptions & Limitations",
        "",
        "1. **Elo ratings** computed from all results since 1872 using tournament-weighted K-factors.",
        "2. **Dixon-Coles model** fitted on 2000–2026 international matches with time-decay weighting.",
        "3. **All WC matches treated as neutral venue** (except USA/Canada/Mexico get 50% home boost).",
        "4. **Red cards always predicted as 0** — modal value is 0 in ~78% of WC matches.",
        "5. **Playoff slots** (6 undetermined qualifiers) use median Elo/FIFA points of likely candidates.",
        "6. **Knockout simulation** uses 50,000 iterations with Dixon-Coles score sampling.",
        "7. **group_fixtures.csv** derived from schedule_2026.csv; venues are approximate.",
        "8. **Best-3rd rules**: 8 of 12 third-placed teams advance, ranked by pts→GD→GF→discipline.",
        "",
        "---",
        "*Pipeline: Elo + Dixon-Coles (60%) + XGBoost (40%) ensemble, calibrated on 2022 WC holdout.*",
    ]

    content = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(content)
    logger.info(f"Saved SUMMARY.md → {output_path}")
    return content
