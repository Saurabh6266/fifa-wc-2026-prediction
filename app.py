"""
app.py — Gradio web app for the FIFA WC 2026 ML Prediction Pipeline.
Deploy to: Hugging Face Spaces (Gradio SDK) — free, permanent, no credit card needed.
"""

import gradio as gr
import pandas as pd
from pathlib import Path

# ── Load pre-computed predictions ────────────────────────────────────────────
BASE = Path(__file__).parent
PRED = BASE / "outputs" / "predictions"

def load_file(filename, is_csv=True):
    """Helper to load files whether they are nested or in the root directory."""
    # Check nested path first (local repo structure)
    nested_path = PRED / filename
    if nested_path.exists():
        return pd.read_csv(nested_path) if is_csv else nested_path.read_text()
    
    # Check data/processed for tournament_winner_probs.csv
    data_path = BASE / "data" / "processed" / filename
    if data_path.exists():
        return pd.read_csv(data_path) if is_csv else data_path.read_text()

    # Check root directory (if user uploaded flat files to HF Spaces)
    flat_path = BASE / filename
    if flat_path.exists():
        return pd.read_csv(flat_path) if is_csv else flat_path.read_text()
    
    raise FileNotFoundError(f"Could not find {filename} in {PRED}, {data_path.parent}, or {BASE}")

group_preds    = load_file("group_stage_predictions.csv")
knockout_preds = load_file("knockout_predictions.csv")
cc_preds       = load_file("corners_cards_predictions.csv")
summary_md     = load_file("SUMMARY.md", is_csv=False)

try:
    winner_probs = load_file("tournament_winner_probs.csv")
except Exception:
    winner_probs = pd.DataFrame()

# ── Helper formatters ─────────────────────────────────────────────────────────
def format_group_preds():
    cols = ["match_id", "group", "date_utc", "home_team", "away_team",
            "predicted_outcome", "exp_home_goals", "exp_away_goals",
            "p_home_win", "p_draw", "p_away_win"]
    df = group_preds[[c for c in cols if c in group_preds.columns]].copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"]).dt.strftime("%b %d")
    df = df.rename(columns={
        "match_id": "#", "group": "Group", "date_utc": "Date",
        "home_team": "Home", "away_team": "Away",
        "predicted_outcome": "Predicted Outcome",
        "exp_home_goals": "xG Home", "exp_away_goals": "xG Away",
        "p_home_win": "P(Home)", "p_draw": "P(Draw)", "p_away_win": "P(Away)"
    })
    return df

def format_knockout_preds():
    cols = ["match_id", "round", "date_utc", "predicted_home", "predicted_away",
            "match_winner", "home_win_prob", "away_win_prob"]
    df = knockout_preds[[c for c in cols if c in knockout_preds.columns]].copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"]).dt.strftime("%b %d")
    df = df.rename(columns={
        "match_id": "#", "round": "Round", "date_utc": "Date",
        "predicted_home": "Team A", "predicted_away": "Team B",
        "match_winner": "Predicted Winner",
        "home_win_prob": "P(Team A)", "away_win_prob": "P(Team B)"
    })
    return df

def format_winner_probs():
    if winner_probs.empty:
        return pd.DataFrame({"Note": ["winner_probs.csv not found"]})
    df = winner_probs.head(20).copy()
    df["win_probability"] = df["win_probability"].map(lambda x: f"{x:.1%}")
    df["finalist_prob"] = df["finalist_prob"].map(lambda x: f"{x:.1%}")
    df["semi_prob"] = df["semi_prob"].map(lambda x: f"{x:.1%}")
    df["quarter_prob"] = df["quarter_prob"].map(lambda x: f"{x:.1%}")
    df = df.rename(columns={
        "team": "Team", "win_probability": "🏆 Win Prob",
        "finalist_prob": "🥈 Final Prob", "semi_prob": "🥉 Semi Prob",
        "quarter_prob": "QF Prob"
    })
    return df

def search_team(team_name: str):
    if not team_name.strip():
        return "Enter a team name to search.", pd.DataFrame(), pd.DataFrame()
    
    name = team_name.strip().title()
    
    # Group stage matches
    gs = group_preds[
        group_preds["home_team"].str.contains(name, case=False, na=False) |
        group_preds["away_team"].str.contains(name, case=False, na=False)
    ].copy()
    
    # Knockout matches
    ko = knockout_preds[
        knockout_preds.get("predicted_home", pd.Series()).str.contains(name, case=False, na=False) |
        knockout_preds.get("predicted_away", pd.Series()).str.contains(name, case=False, na=False)
    ].copy() if "predicted_home" in knockout_preds.columns else pd.DataFrame()
    
    # Win prob
    wp = ""
    if not winner_probs.empty:
        match = winner_probs[winner_probs["team"].str.contains(name, case=False, na=False)]
        if len(match):
            r = match.iloc[0]
            wp = (f"**{r['team']}** — Tournament Win Probability: **{r['win_probability']:.1%}**\n\n"
                  f"- Reach Final: {r['finalist_prob']:.1%}\n"
                  f"- Reach Semi-final: {r['semi_prob']:.1%}\n"
                  f"- Reach Quarter-final: {r['quarter_prob']:.1%}")
        else:
            wp = f"*{name}* not found in winner probability data (may not have qualified for knockout)."
    
    gs_display = gs[["group", "date_utc", "home_team", "away_team", 
                      "predicted_outcome", "exp_home_goals", "exp_away_goals",
                      "p_home_win", "p_draw", "p_away_win"]].rename(columns={
                          "group": "Group", "date_utc": "Date",
                          "home_team": "Home", "away_team": "Away",
                          "predicted_outcome": "Outcome",
                          "exp_home_goals": "xG Home", "exp_away_goals": "xG Away"
                      }) if len(gs) else pd.DataFrame({"Note": [f"No group stage matches found for {name}"]})
    
    ko_display = ko[["round", "predicted_home", "predicted_away", "match_winner",
                     "home_win_prob", "away_win_prob"]].rename(columns={
                         "round": "Round", "predicted_home": "Team A",
                         "predicted_away": "Team B", "match_winner": "Predicted Winner",
                         "home_win_prob": "P(A)", "away_win_prob": "P(B)"
                     }) if len(ko) else pd.DataFrame({"Note": [f"No knockout predictions yet for {name}"]})
    
    return wp, gs_display, ko_display


# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui"]
    ),
    title="FIFA WC 2026 ML Predictions",
    css="""
        .gr-prose h1 { font-size: 2em; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    # 🏆 FIFA World Cup 2026 — ML Prediction Dashboard
    **Powered by Dixon-Coles + Elo Rating + XGBoost | 50,000 Monte Carlo simulations**
    
    > **Model accuracy: 56.5%** on 69 real group-stage matches (vs 43.5% naive baseline)
    """)

    with gr.Tabs():

        # Tab 1 — Summary Report
        with gr.Tab("📋 Prediction Summary"):
            gr.Markdown(summary_md)

        # Tab 2 — Tournament Winner Probabilities
        with gr.Tab("🏆 Winner Probabilities"):
            gr.Markdown("### Top 20 teams most likely to win the 2026 FIFA World Cup")
            gr.Dataframe(
                value=format_winner_probs(),
                interactive=False,
                wrap=True,
            )

        # Tab 3 — Group Stage Predictions
        with gr.Tab("⚽ Group Stage (72 matches)"):
            gr.Markdown("### All 72 group stage match predictions with expected goals and win probabilities")
            gr.Dataframe(
                value=format_group_preds(),
                interactive=False,
                wrap=True,
            )

        # Tab 4 — Knockout Bracket
        with gr.Tab("🥊 Knockout Bracket"):
            gr.Markdown("### Predicted knockout bracket — most likely teams in each slot")
            gr.Dataframe(
                value=format_knockout_preds(),
                interactive=False,
                wrap=True,
            )

        # Tab 5 — Search by Team
        with gr.Tab("🔍 Search by Team"):
            gr.Markdown("### Look up all predictions for a specific team")
            with gr.Row():
                team_input = gr.Textbox(
                    label="Team Name",
                    placeholder="e.g. Argentina, Brazil, Japan...",
                    scale=3
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)
            
            wp_output = gr.Markdown(label="Tournament Win Probability")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Group Stage Matches")
                    gs_table = gr.Dataframe(interactive=False, wrap=True)
                with gr.Column():
                    gr.Markdown("#### Knockout Stage Predictions")
                    ko_table = gr.Dataframe(interactive=False, wrap=True)
            
            search_btn.click(
                fn=search_team,
                inputs=[team_input],
                outputs=[wp_output, gs_table, ko_table]
            )
            team_input.submit(
                fn=search_team,
                inputs=[team_input],
                outputs=[wp_output, gs_table, ko_table]
            )

        # Tab 6 — Methodology
        with gr.Tab("🔬 Methodology"):
            gr.Markdown("""
## How This Works

### Models Used
1. **Dixon-Coles Bivariate Poisson** — Core model. Fits attack/defense parameters for every team using 
   time-decay weighting on ~3,000 matches since 2015. Models low-score correction (rho parameter).
   
2. **Elo Rating System** — Complementary team strength metric computed from all international results 
   since 1872 with tournament-based K-factors (WC=60, Confederations=40, Friendlies=20).
   
3. **XGBoost Classifier** — Trained on 25+ engineered features (rolling form, Elo delta, FIFA ranking, 
   head-to-head) to predict Home Win / Draw / Away Win outcome probabilities.

4. **Monte Carlo Simulation** — 50,000 full tournament simulations to generate bracket progression 
   probabilities. Each simulation independently samples scorelines from the Dixon-Coles distribution.

### Feature Engineering
- 5-match rolling goal average (home & away)
- 10-match rolling win rate
- Elo rating difference
- FIFA ranking difference  
- Head-to-head record (last 5 meetings)
- Tournament stage weighting

### Validation
- Backtest on **72 real 2026 WC group-stage matches**
- **56.5% outcome accuracy** vs 43.5% naive FIFA-rank baseline

### Data Sources
- [Kaggle: International football results 1872–2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- [Kaggle: Club Football / FIFA Rankings](https://www.kaggle.com/datasets/...)
- FIFA official 2026 WC schedule
            """)

    gr.Markdown("""
    ---
    Built by **Saurabh Gupta** · [GitHub](https://github.com/Saurabh6266/fifa-wc-2026-prediction) · 
    Model accuracy: **56.5%** on live 2026 WC data
    """)

if __name__ == "__main__":
    demo.launch()
