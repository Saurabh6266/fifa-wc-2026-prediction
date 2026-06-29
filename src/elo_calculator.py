"""
src/elo_calculator.py

Phase 3.1: National Team Elo Rating System.

Builds Elo ratings for all national teams by iterating chronologically
through results.csv. Uses tournament-weighted K-factors and a home
advantage boost. Produces final ratings as of June 11, 2026 (WC start).

IMPORTANT: Do NOT use EloRatings.csv (club Elo) — ratings are computed
from scratch using international match results only.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ELO CONFIGURATION
# ---------------------------------------------------------------------------

INITIAL_ELO = 1500.0
HOME_BOOST = 100.0          # points added to home team for E_home calculation
NEUTRAL_BOOST = 50.0        # reduced boost at neutral venues

# K-factors by tournament importance
K_FACTORS = {
    "world_cup":    60,
    "continental":  50,
    "wc_qualifier": 40,
    "minor_tourney": 35,
    "friendly":     30,
}

# Tournament name → K-factor category
def _get_k_factor(tournament: str, neutral: bool) -> int:
    """
    Determine K-factor from tournament name string.

    Parameters
    ----------
    tournament : str
        Tournament name from results.csv.
    neutral : bool
        Whether match was played at a neutral venue.

    Returns
    -------
    int
        K-factor for this match type.
    """
    t = str(tournament).lower()

    if "fifa world cup" in t and "qualification" not in t:
        return K_FACTORS["world_cup"]

    if any(kw in t for kw in ["copa america", "euro ", "european championship",
                               "africa cup", "african cup", "asian cup",
                               "gold cup", "nations cup", "concacaf nations",
                               "ofc nations"]):
        return K_FACTORS["continental"]

    if "qualif" in t or "qualifier" in t or "qualification" in t:
        return K_FACTORS["wc_qualifier"]

    if "friendly" in t or "international" in t:
        return K_FACTORS["friendly"]

    return K_FACTORS["minor_tourney"]


def _win_probability(elo_home: float, elo_away: float, boost: float) -> float:
    """
    Compute expected win probability for the home team using Elo formula.

    E_home = 1 / (1 + 10^((elo_away - elo_home - boost) / 400))

    Parameters
    ----------
    elo_home : float
    elo_away : float
    boost : float
        Home advantage in Elo points (0 = perfectly neutral).

    Returns
    -------
    float
        P(home wins), in [0, 1].
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home - boost) / 400.0))


# ---------------------------------------------------------------------------
# ELO CALCULATOR CLASS
# ---------------------------------------------------------------------------

class EloCalculator:
    """
    Iterative Elo rating system for national football teams.

    Usage
    -----
    elo = EloCalculator()
    elo.fit(results_df)                  # build ratings from match history
    ratings = elo.get_ratings_at(date)   # get snapshot at any point
    elo.save(path)                       # save to CSV
    """

    def __init__(self, initial_elo: float = INITIAL_ELO):
        self.initial_elo = initial_elo
        self.ratings: dict = {}           # current ratings {team: elo}
        self.history: list = []           # list of (date, team, elo) snapshots
        self._ratings_at_date: dict = {}  # cache for date-indexed ratings
        self._fitted = False

    def _get_or_init(self, team: str) -> float:
        """Return current Elo for a team, initialising at default if unseen."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_elo
        return self.ratings[team]

    def fit(self, results_df: pd.DataFrame) -> "EloCalculator":
        """
        Build Elo ratings by iterating chronologically through all matches.

        Parameters
        ----------
        results_df : pd.DataFrame
            Full results DataFrame (already name-normalized and date-parsed).
            Required columns: date, home_team, away_team, home_score,
                              away_score, tournament, neutral.

        Returns
        -------
        self
        """
        df = results_df.sort_values("date").reset_index(drop=True)
        logger.info(f"Fitting Elo on {len(df):,} matches ({df['date'].min().year}–{df['date'].max().year})")

        self.history = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            h_score = row["home_score"]
            a_score = row["away_score"]
            neutral = bool(row.get("neutral", False))

            elo_h = self._get_or_init(home)
            elo_a = self._get_or_init(away)

            boost = NEUTRAL_BOOST if neutral else HOME_BOOST
            k = _get_k_factor(row.get("tournament", ""), neutral)

            e_home = _win_probability(elo_h, elo_a, boost)
            e_away = 1.0 - e_home

            # Actual outcome (1=win, 0.5=draw, 0=loss)
            if h_score > a_score:
                s_home, s_away = 1.0, 0.0
            elif h_score == a_score:
                s_home, s_away = 0.5, 0.5
            else:
                s_home, s_away = 0.0, 1.0

            # Update ratings
            self.ratings[home] = elo_h + k * (s_home - e_home)
            self.ratings[away] = elo_a + k * (s_away - e_away)

            # Record snapshot
            self.history.append({
                "date":  row["date"],
                "team":  home,
                "elo":   self.ratings[home],
                "match": f"{home} vs {away}",
            })
            self.history.append({
                "date":  row["date"],
                "team":  away,
                "elo":   self.ratings[away],
                "match": f"{home} vs {away}",
            })

        self._fitted = True
        logger.info(f"Elo fitting complete. Tracked {len(self.ratings)} teams.")
        self._validate_top_teams()
        return self

    def _validate_top_teams(self):
        """Warn if top WC favourites are not in top 8 Elo — indicates a bug."""
        expected_top = {"France", "Argentina", "Brazil", "Spain", "England"}
        top8 = set(sorted(self.ratings, key=self.ratings.get, reverse=True)[:8])
        missing = expected_top - top8
        if missing:
            logger.warning(
                f"VALIDATION: Expected top teams not in top-8 Elo: {missing}. "
                "Check name normalization!"
            )
        else:
            logger.info("VALIDATION: All expected top teams in top-8 Elo ✓")

    def get_rating(self, team: str, default: float = None) -> float:
        """
        Get the current (post-fit) Elo rating for a team.

        Parameters
        ----------
        team : str
            Canonical team name.
        default : float, optional
            Value to return if team not found. Defaults to initial_elo.

        Returns
        -------
        float
        """
        return self.ratings.get(team, default if default is not None else self.initial_elo)

    def get_ratings_as_of(self, cutoff_date: pd.Timestamp) -> dict:
        """
        Return the Elo rating for each team as of a specific date.
        Uses the most recent snapshot ≤ cutoff_date for each team.

        Parameters
        ----------
        cutoff_date : pd.Timestamp

        Returns
        -------
        dict
            {team_name: elo_rating}
        """
        if not self.history:
            logger.warning("No history available; returning current ratings.")
            return self.ratings.copy()

        hist_df = pd.DataFrame(self.history)
        hist_df = hist_df[hist_df["date"] <= cutoff_date]

        if hist_df.empty:
            return {}

        latest = hist_df.sort_values("date").groupby("team")["elo"].last()
        return latest.to_dict()

    def get_elo_before_match(self, home: str, away: str, match_date: pd.Timestamp,
                              history_df: pd.DataFrame = None) -> tuple:
        """
        Return Elo ratings for both teams immediately BEFORE a given match.
        Used for building the feature matrix without data leakage.

        Parameters
        ----------
        home : str
        away : str
        match_date : pd.Timestamp
        history_df : pd.DataFrame, optional
            Pre-built history dataframe (for efficiency in loops).

        Returns
        -------
        (float, float)
            (elo_home, elo_away) at match_date - 1 day
        """
        cutoff = match_date - pd.Timedelta(days=1)
        ratings = self.get_ratings_as_of(cutoff)
        return (ratings.get(home, self.initial_elo),
                ratings.get(away, self.initial_elo))

    def save(self, path=None):
        """
        Save final Elo ratings to CSV.

        Parameters
        ----------
        path : str or Path, optional
            Output path. Defaults to data/processed/team_elo_ratings.csv.
        """
        from pathlib import Path
        if path is None:
            path = Path(__file__).parent.parent / "data" / "processed" / "team_elo_ratings.csv"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([
            {"team": t, "elo_rating": r}
            for t, r in sorted(self.ratings.items(), key=lambda x: -x[1])
        ])
        df.to_csv(path, index=False)
        logger.info(f"Saved Elo ratings to {path} ({len(df)} teams)")
        return df

    def to_history_df(self) -> pd.DataFrame:
        """Return full Elo history as a DataFrame."""
        return pd.DataFrame(self.history)

    def get_top_n(self, n: int = 20) -> pd.DataFrame:
        """Return top-N teams by current Elo rating."""
        df = pd.DataFrame(
            [{"rank": i+1, "team": t, "elo": r}
             for i, (t, r) in enumerate(
                 sorted(self.ratings.items(), key=lambda x: -x[1])[:n]
             )]
        )
        return df


# ---------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ---------------------------------------------------------------------------

def build_elo_ratings(results_df: pd.DataFrame,
                      cutoff_date: pd.Timestamp = pd.Timestamp("2026-06-10"),
                      save: bool = True) -> tuple:
    """
    Full pipeline: fit Elo on results_df and return (calculator, ratings_dict).

    Parameters
    ----------
    results_df : pd.DataFrame
        All results (name-normalized, date-parsed).
    cutoff_date : pd.Timestamp
        Date to extract final ratings (day before tournament).
    save : bool
        Whether to save ratings CSV.

    Returns
    -------
    (EloCalculator, dict)
        The fitted calculator and a {team: elo} dict as of cutoff_date.
    """
    calc = EloCalculator()
    calc.fit(results_df)

    final_ratings = calc.get_ratings_as_of(cutoff_date)
    logger.info(f"Extracted {len(final_ratings)} team ratings as of {cutoff_date.date()}")

    if save:
        from pathlib import Path
        path = Path(__file__).parent.parent / "data" / "processed" / "team_elo_ratings.csv"
        df = pd.DataFrame([
            {"team": t, "elo_rating": r}
            for t, r in sorted(final_ratings.items(), key=lambda x: -x[1])
        ])
        df.to_csv(path, index=False)
        logger.info(f"Saved final Elo ratings: {path}")

    return calc, final_ratings
