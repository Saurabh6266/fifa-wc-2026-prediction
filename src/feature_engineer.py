"""
src/feature_engineer.py

Phase 3.2–3.3: Match Feature Matrix Construction.

Builds the feature matrix for training and WC 2026 prediction.
All features are computed using only pre-match information (no leakage).

Features:
- Elo ratings (before each match)
- FIFA points and ranking
- Rolling form (last 5/10 matches, weighted by recency)
- Rolling average goals scored/conceded
- Head-to-head record (last 5 meetings)
- Match context (neutral, WC, knockout)
- Confederation pair encoding
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Confederation numerical encodings (for model features)
CONF_ENCODE = {
    "UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "AFC": 3, "CAF": 4, "OFC": 5
}


# ---------------------------------------------------------------------------
# ROLLING STATS BUILDER
# ---------------------------------------------------------------------------

class RollingStatsEngine:
    """
    Computes rolling match statistics for each team with strict no-leakage
    guarantees: each match's features use only matches BEFORE that date.
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Parameters
        ----------
        results_df : pd.DataFrame
            Sorted chronologically. Columns: date, home_team, away_team,
            home_score, away_score, neutral.
        """
        self.df = results_df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
        self._build_team_match_index()

    def _build_team_match_index(self):
        """Pre-build a per-team chronological list of matches for O(1) lookup."""
        self._team_matches = defaultdict(list)  # {team: [(date, is_home, h_goals, a_goals, result), ...]}

        for _, row in self.df.iterrows():
            h, a = row["home_team"], row["away_team"]
            hg, ag = int(row["home_score"]), int(row["away_score"])
            date = row["date"]
            neutral = bool(row.get("neutral", False))

            # Home team
            result_h = "W" if hg > ag else ("D" if hg == ag else "L")
            self._team_matches[h].append({
                "date": date, "is_home": True, "goals_for": hg, "goals_against": ag,
                "result": result_h, "neutral": neutral, "opponent": a
            })

            # Away team
            result_a = "W" if ag > hg else ("D" if ag == hg else "L")
            self._team_matches[a].append({
                "date": date, "is_home": False, "goals_for": ag, "goals_against": hg,
                "result": result_a, "neutral": neutral, "opponent": h
            })

    def get_matches_before(self, team: str, cutoff_date: pd.Timestamp) -> list:
        """Return all matches for team strictly before cutoff_date."""
        return [m for m in self._team_matches.get(team, []) if m["date"] < cutoff_date]

    def form_score(self, team: str, cutoff_date: pd.Timestamp, n: int = 10,
                   decay: float = 0.9) -> float:
        """
        Weighted form score over last n matches (most recent weighted highest).

        Points: Win=3, Draw=1, Loss=0. Weighted by recency (decay^(n-1), ..., decay^0).

        Parameters
        ----------
        team : str
        cutoff_date : pd.Timestamp
        n : int  Number of matches to consider.
        decay : float  Recency decay factor.

        Returns
        -------
        float  Weighted form score, normalised to [0, 1].
        """
        matches = self.get_matches_before(team, cutoff_date)[-n:]
        if not matches:
            return 0.5  # neutral prior

        pts = {"W": 3, "D": 1, "L": 0}
        weights = [decay ** (len(matches) - 1 - i) for i in range(len(matches))]
        score = sum(w * pts[m["result"]] for w, m in zip(weights, matches))
        max_possible = sum(3 * w for w in weights)
        return score / max_possible if max_possible > 0 else 0.5

    def avg_goals(self, team: str, cutoff_date: pd.Timestamp, n: int = 10,
                  for_or_against: str = "for") -> float:
        """
        Rolling average goals scored/conceded over last n matches.

        Parameters
        ----------
        for_or_against : str  'for' (scored) or 'against' (conceded).

        Returns
        -------
        float  Average goals, or global mean prior (1.2) if no data.
        """
        matches = self.get_matches_before(team, cutoff_date)[-n:]
        if not matches:
            return 1.2  # global prior ~1.2 goals/match

        key = "goals_for" if for_or_against == "for" else "goals_against"
        return np.mean([m[key] for m in matches])

    def days_since_last_match(self, team: str, cutoff_date: pd.Timestamp) -> float:
        """Days since last match before cutoff_date. Returns 30 if no history."""
        matches = self.get_matches_before(team, cutoff_date)
        if not matches:
            return 30.0
        last_date = matches[-1]["date"]
        return float((cutoff_date - last_date).days)

    def h2h_record(self, home: str, away: str, cutoff_date: pd.Timestamp,
                   n: int = 5) -> tuple:
        """
        Head-to-head record between home and away teams in last n meetings.

        Parameters
        ----------
        Returns
        -------
        (home_wins, draws, away_wins)
        """
        home_matches = self.get_matches_before(home, cutoff_date)
        h2h = [m for m in home_matches if m["opponent"] == away][-n:]
        hw = sum(1 for m in h2h if m["result"] == "W")
        draws = sum(1 for m in h2h if m["result"] == "D")
        aw = sum(1 for m in h2h if m["result"] == "L")
        return hw, draws, aw


# ---------------------------------------------------------------------------
# FEATURE MATRIX BUILDER
# ---------------------------------------------------------------------------

def build_match_features(results_df: pd.DataFrame,
                          elo_calculator,
                          fifa_rankings: dict,
                          confederation_map: dict,
                          min_year: int = 1990,
                          save: bool = True) -> pd.DataFrame:
    """
    Build the full feature matrix for all historical matches (1990+).

    Features use only pre-match data (no leakage).

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results (post-2000 recommended for model training).
    elo_calculator : EloCalculator
        Fitted Elo calculator.
    fifa_rankings : dict
        {team: {rank, points, confederation}} for WC teams.
    confederation_map : dict
        {team: confederation_string} — fallback for non-WC teams.
    min_year : int
        Only build features for matches from this year onwards.
    save : bool
        Save to data/processed/match_features.csv.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per match.
    """
    df = results_df[results_df["date"].dt.year >= min_year].copy()
    df = df.dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    logger.info(f"Building features for {len(df):,} matches ({min_year}+)")

    engine = RollingStatsEngine(results_df)  # use all history for rolling stats
    elo_history = pd.DataFrame(elo_calculator.history) if elo_calculator.history else None

    rows = []
    for i, row in df.iterrows():
        if i % 5000 == 0:
            logger.info(f"  Processing match {i:,}/{len(df):,}")

        home  = row["home_team"]
        away  = row["away_team"]
        date  = row["date"]
        neutral = bool(row.get("neutral", False))

        # Elo (before match)
        elo_h = elo_calculator.get_ratings_as_of(date - pd.Timedelta(days=1)).get(home, 1500)
        elo_a = elo_calculator.get_ratings_as_of(date - pd.Timedelta(days=1)).get(away, 1500)

        # FIFA data
        fi_h = fifa_rankings.get(home, {})
        fi_a = fifa_rankings.get(away, {})

        # Form
        form_h5  = engine.form_score(home, date, n=5)
        form_h10 = engine.form_score(home, date, n=10)
        form_a5  = engine.form_score(away, date, n=5)
        form_a10 = engine.form_score(away, date, n=10)

        # Goals
        gf_h = engine.avg_goals(home, date, n=10, for_or_against="for")
        gc_h = engine.avg_goals(home, date, n=10, for_or_against="against")
        gf_a = engine.avg_goals(away, date, n=10, for_or_against="for")
        gc_a = engine.avg_goals(away, date, n=10, for_or_against="against")

        # H2H
        h2h_hw, h2h_d, h2h_aw = engine.h2h_record(home, away, date, n=5)

        # Freshness
        days_h = engine.days_since_last_match(home, date)
        days_a = engine.days_since_last_match(away, date)

        # Confederation
        conf_h = (fi_h.get("confederation") or
                  confederation_map.get(home, "Unknown"))
        conf_a = (fi_a.get("confederation") or
                  confederation_map.get(away, "Unknown"))

        # Match context
        is_wc = bool(row.get("is_wc", False))
        is_knockout = bool(row.get("is_knockout", False))

        # Target
        hg, ag = int(row["home_score"]), int(row["away_score"])
        if hg > ag:
            outcome = 2   # home win
        elif hg == ag:
            outcome = 1   # draw
        else:
            outcome = 0   # away win

        rows.append({
            # Match metadata
            "date":             date,
            "home_team":        home,
            "away_team":        away,
            "tournament":       row.get("tournament", ""),
            "is_neutral":       int(neutral),
            "is_wc_match":      int(is_wc),
            "is_knockout":      int(is_knockout),
            # Elo
            "elo_home":         round(elo_h, 1),
            "elo_away":         round(elo_a, 1),
            "elo_diff":         round(elo_h - elo_a, 1),
            "elo_ratio":        round(elo_h / max(elo_a, 1), 4),
            # FIFA rankings (NaN for non-WC teams)
            "fifa_points_home": fi_h.get("points", np.nan),
            "fifa_rank_home":   fi_h.get("rank", np.nan),
            "fifa_points_away": fi_a.get("points", np.nan),
            "fifa_rank_away":   fi_a.get("rank", np.nan),
            "fifa_points_diff": (fi_h.get("points", np.nan) if fi_h else np.nan) -
                                (fi_a.get("points", np.nan) if fi_a else np.nan),
            # Form
            "form_home_5":      round(form_h5, 4),
            "form_home_10":     round(form_h10, 4),
            "form_away_5":      round(form_a5, 4),
            "form_away_10":     round(form_a10, 4),
            "form_diff_10":     round(form_h10 - form_a10, 4),
            # Goals
            "goals_scored_avg_home":    round(gf_h, 3),
            "goals_conceded_avg_home":  round(gc_h, 3),
            "goals_scored_avg_away":    round(gf_a, 3),
            "goals_conceded_avg_away":  round(gc_a, 3),
            "attack_diff":      round(gf_h - gc_a, 3),   # home attack vs away defense
            "defense_diff":     round(gf_a - gc_h, 3),   # away attack vs home defense
            # H2H
            "h2h_home_wins_5":  h2h_hw,
            "h2h_draws_5":      h2h_d,
            "h2h_away_wins_5":  h2h_aw,
            "h2h_diff":         h2h_hw - h2h_aw,
            # Freshness
            "days_since_last_home": days_h,
            "days_since_last_away": days_a,
            # Confederation
            "conf_home":        CONF_ENCODE.get(conf_h, -1),
            "conf_away":        CONF_ENCODE.get(conf_a, -1),
            "conf_home_str":    conf_h,
            "conf_away_str":    conf_a,
            # Targets
            "home_goals":       hg,
            "away_goals":       ag,
            "total_goals":      hg + ag,
            "goal_diff":        hg - ag,
            "outcome":          outcome,
        })

    feat_df = pd.DataFrame(rows)

    if save:
        out = DATA_PROCESSED / "match_features.csv"
        feat_df.to_csv(out, index=False)
        logger.info(f"Saved match_features.csv: {len(feat_df):,} rows → {out}")

    return feat_df


def build_wc2026_features(group_fixtures_df: pd.DataFrame,
                           elo_ratings: dict,
                           fifa_rankings: dict,
                           confederation_map: dict,
                           rolling_engine: RollingStatsEngine,
                           cutoff_date: pd.Timestamp = pd.Timestamp("2026-06-10"),
                           save: bool = True) -> pd.DataFrame:
    """
    Build feature matrix for all 72 WC 2026 group stage fixtures.

    Parameters
    ----------
    group_fixtures_df : pd.DataFrame
    elo_ratings : dict  {team: elo} as of cutoff_date.
    fifa_rankings : dict
    confederation_map : dict
    rolling_engine : RollingStatsEngine  Fitted on historical results.
    cutoff_date : pd.Timestamp  Date to use for Elo/form features.
    save : bool

    Returns
    -------
    pd.DataFrame
    """
    rows = []

    for _, fix in group_fixtures_df.iterrows():
        home = fix["home_team"]
        away = fix["away_team"]
        is_placeholder = ("Playoff" in home or "Playoff" in away)

        elo_h = elo_ratings.get(home, 1500.0)
        elo_a = elo_ratings.get(away, 1500.0)
        fi_h = fifa_rankings.get(home, {})
        fi_a = fifa_rankings.get(away, {})
        conf_h = fi_h.get("confederation") or confederation_map.get(home, "Unknown")
        conf_a = fi_a.get("confederation") or confederation_map.get(away, "Unknown")

        form_h5  = rolling_engine.form_score(home, cutoff_date, n=5)
        form_h10 = rolling_engine.form_score(home, cutoff_date, n=10)
        form_a5  = rolling_engine.form_score(away, cutoff_date, n=5)
        form_a10 = rolling_engine.form_score(away, cutoff_date, n=10)

        gf_h = rolling_engine.avg_goals(home, cutoff_date, n=10, for_or_against="for")
        gc_h = rolling_engine.avg_goals(home, cutoff_date, n=10, for_or_against="against")
        gf_a = rolling_engine.avg_goals(away, cutoff_date, n=10, for_or_against="for")
        gc_a = rolling_engine.avg_goals(away, cutoff_date, n=10, for_or_against="against")

        h2h_hw, h2h_d, h2h_aw = rolling_engine.h2h_record(home, away, cutoff_date)

        rows.append({
            "match_id":         fix["match_id"],
            "group":            fix["group"],
            "date_utc":         fix["date_utc"],
            "venue":            fix.get("venue", ""),
            "home_team":        home,
            "away_team":        away,
            "is_placeholder":   int(is_placeholder),
            "is_neutral":       1,   # All WC group stage = neutral for non-hosts
            "is_wc_match":      1,
            "is_knockout":      0,
            "elo_home":         round(elo_h, 1),
            "elo_away":         round(elo_a, 1),
            "elo_diff":         round(elo_h - elo_a, 1),
            "elo_ratio":        round(elo_h / max(elo_a, 1), 4),
            "fifa_points_home": fi_h.get("points", np.nan),
            "fifa_rank_home":   fi_h.get("rank", np.nan),
            "fifa_points_away": fi_a.get("points", np.nan),
            "fifa_rank_away":   fi_a.get("rank", np.nan),
            "fifa_points_diff": (fi_h.get("points", np.nan) if fi_h else np.nan) -
                                (fi_a.get("points", np.nan) if fi_a else np.nan),
            "form_home_5":      round(form_h5, 4),
            "form_home_10":     round(form_h10, 4),
            "form_away_5":      round(form_a5, 4),
            "form_away_10":     round(form_a10, 4),
            "form_diff_10":     round(form_h10 - form_a10, 4),
            "goals_scored_avg_home":   round(gf_h, 3),
            "goals_conceded_avg_home": round(gc_h, 3),
            "goals_scored_avg_away":   round(gf_a, 3),
            "goals_conceded_avg_away": round(gc_a, 3),
            "attack_diff":      round(gf_h - gc_a, 3),
            "defense_diff":     round(gf_a - gc_h, 3),
            "h2h_home_wins_5":  h2h_hw,
            "h2h_draws_5":      h2h_d,
            "h2h_away_wins_5":  h2h_aw,
            "h2h_diff":         h2h_hw - h2h_aw,
            "conf_home":        CONF_ENCODE.get(conf_h, -1),
            "conf_away":        CONF_ENCODE.get(conf_a, -1),
            "conf_home_str":    conf_h,
            "conf_away_str":    conf_a,
        })

    wc_feat = pd.DataFrame(rows)

    if save:
        out = DATA_PROCESSED / "wc2026_features.csv"
        wc_feat.to_csv(out, index=False)
        logger.info(f"Saved wc2026_features.csv: {len(wc_feat)} rows → {out}")

    return wc_feat


# ---------------------------------------------------------------------------
# XGBOOST FEATURE COLUMNS (subset used for training the classifier)
# ---------------------------------------------------------------------------

XGB_FEATURE_COLS = [
    "elo_diff", "elo_ratio",
    "form_diff_10", "form_home_10", "form_away_10",
    "attack_diff", "defense_diff",
    "goals_scored_avg_home", "goals_conceded_avg_home",
    "goals_scored_avg_away", "goals_conceded_avg_away",
    "h2h_diff", "h2h_home_wins_5",
    "is_neutral", "is_wc_match", "is_knockout",
    "conf_home", "conf_away",
    "days_since_last_home", "days_since_last_away",
]

XGB_FIFA_COLS = ["fifa_points_diff"]   # Only when available (WC matches)
