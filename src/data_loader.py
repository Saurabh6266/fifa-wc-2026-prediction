"""
src/data_loader.py

Phase 1: Data loading, cleaning, and team registry construction.

Loads all raw data files, applies name normalization, validates data quality,
and produces the master team_registry.csv used throughout the pipeline.
"""

import os
import logging
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

from src.name_normalizer import (
    normalize_dataframe_teams,
    build_former_names_map,
    FIFA_RANKINGS_MAP,
    RESULTS_CSV_MAP,
    NORMALIZATION_MAP,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

CUTOFF_DATE = pd.Timestamp("2026-06-11")  # Tournament start date

# ---------------------------------------------------------------------------
# KNOWN CONFEDERATION MAPPING (for teams missing from FIFA_Rankings.csv)
# ---------------------------------------------------------------------------
CONFEDERATION_MAP = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Uruguay": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Peru": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "England": "UEFA", "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA",
    "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA",
    "Denmark": "UEFA", "Norway": "UEFA", "Sweden": "UEFA", "Scotland": "UEFA",
    "Serbia": "UEFA", "Czechia": "UEFA", "Slovakia": "UEFA", "Poland": "UEFA",
    "Hungary": "UEFA", "Türkiye": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Ukraine": "UEFA", "Wales": "UEFA", "Romania": "UEFA", "Greece": "UEFA",
    "Albania": "UEFA", "Slovenia": "UEFA", "Georgia": "UEFA",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC", "Iran": "AFC",
    "Saudi Arabia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "Jordan": "AFC",
    "Uzbekistan": "AFC", "Oman": "AFC", "Syria": "AFC", "UAE": "AFC",
    "New Zealand": "OFC",
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Algeria": "CAF",
    "Tunisia": "CAF", "Ghana": "CAF", "Côte d'Ivoire": "CAF", "Nigeria": "CAF",
    "South Africa": "CAF", "Cameroon": "CAF", "Congo DR": "CAF",
    "Cabo Verde": "CAF", "Mali": "CAF", "Burkina Faso": "CAF",
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Panama": "CONCACAF", "Curaçao": "CONCACAF", "Haiti": "CONCACAF",
    "Jamaica": "CONCACAF", "Trinidad & Tobago": "CONCACAF", "Honduras": "CONCACAF",
    "Costa Rica": "CONCACAF", "El Salvador": "CONCACAF", "Guatemala": "CONCACAF",
}

CONF_WEIGHT = {"UEFA": 1.0, "CONMEBOL": 1.0, "CONCACAF": 0.85,
               "AFC": 0.85, "CAF": 0.85, "OFC": 0.85}

# Host nations (get slight home boost in group stage)
HOST_NATIONS = {"USA", "Canada", "Mexico"}


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_results(path=None, min_year=1990):
    """
    Load and clean results.csv (international match results).

    Parameters
    ----------
    path : str or Path, optional
        Path to results.csv. Defaults to data/raw/results.csv.
    min_year : int
        Minimum year to include in primary modeling set (default 1990).

    Returns
    -------
    pd.DataFrame
        Cleaned results dataframe with normalized team names.
        Extra columns: is_wc, is_knockout, year.
    """
    path = path or DATA_RAW / "results.csv"
    logger.info(f"Loading results from {path}")

    df = pd.read_csv(path, parse_dates=["date"])
    logger.info(f"  Raw rows: {len(df):,}")

    # Normalize team names — results.csv uses IR Iran, Korea Republic etc.
    normalize_dataframe_teams(df, ["home_team", "away_team"], extra_map=RESULTS_CSV_MAP)
    # Apply general normalization on top
    normalize_dataframe_teams(df, ["home_team", "away_team"])

    # Add derived columns
    df["year"] = df["date"].dt.year
    df["is_wc"] = df["tournament"].str.contains("FIFA World Cup", na=False)
    df["is_knockout"] = df["tournament"].str.contains(
        r"Round of|Quarter|Semi|Final", na=False, regex=True
    ) & df["is_wc"]

    # Validate: no negative scores
    invalid = df[(df["home_score"] < 0) | (df["away_score"] < 0)]
    if len(invalid) > 0:
        logger.warning(f"  Found {len(invalid)} rows with negative scores — dropping")
        df = df[~((df["home_score"] < 0) | (df["away_score"] < 0))]

    # Duplicate check
    dupes = df.duplicated(subset=["date", "home_team", "away_team"])
    if dupes.sum() > 0:
        logger.warning(f"  Found {dupes.sum()} duplicate match entries — dropping")
        df = df[~dupes]

    df_primary = df[df["year"] >= min_year].copy()
    logger.info(f"  Full dataset: {len(df):,} rows | Primary ({min_year}+): {len(df_primary):,}")

    return df, df_primary


def load_former_names(path=None):
    """
    Load former_names.csv and build a historical name-change mapping.

    Parameters
    ----------
    path : str or Path, optional
        Path to former_names.csv.

    Returns
    -------
    dict
        {old_name: current_name}
    """
    path = path or DATA_RAW / "former_names.csv"
    return build_former_names_map(str(path))


def load_group_fixtures(path=None):
    """
    Load group_fixtures.csv — the authoritative group assignments.

    Parameters
    ----------
    path : str or Path, optional

    Returns
    -------
    pd.DataFrame
        Group fixtures with normalized team names.
        Columns: match_id, group, home_team, away_team, date_utc, venue.
    """
    path = path or DATA_RAW / "group_fixtures.csv"
    df = pd.read_csv(path, parse_dates=["date_utc"])
    normalize_dataframe_teams(df, ["home_team", "away_team"])
    logger.info(f"Loaded group fixtures: {len(df)} matches, groups {sorted(df['group'].unique())}")
    return df


def load_knockout_slots(path=None):
    """
    Load knockout_slots.csv — the bracket structure for matches 73-104.

    Parameters
    ----------
    path : str or Path, optional

    Returns
    -------
    pd.DataFrame
        Knockout slot definitions.
    """
    path = path or DATA_RAW / "knockout_slots.csv"
    df = pd.read_csv(path, parse_dates=["date_utc"])
    logger.info(f"Loaded knockout slots: {len(df)} matches")
    return df


def load_fifa_rankings(path=None):
    """
    Load FIFA_Rankings.csv with only confirmed WC 2026 teams.

    Known issues handled:
    - Group column is ignored (groups derived from group_fixtures.csv)
    - Missing teams filled with estimated values
    - Name mismatches normalized

    Parameters
    ----------
    path : str or Path, optional

    Returns
    -------
    dict
        {team_name: {'rank': int, 'points': float, 'confederation': str, 'conf_weight': float}}
    """
    path = path or DATA_RAW / "FIFA_Rankings.csv"
    df = pd.read_csv(path)
    normalize_dataframe_teams(df, ["Team"], extra_map=FIFA_RANKINGS_MAP)

    result = {}
    for _, row in df.iterrows():
        team = row["Team"]
        conf = row["Confederation"]
        result[team] = {
            "rank": int(row["Rank"]),
            "points": float(row["Points"]),
            "confederation": conf,
            "conf_weight": float(row["Conf_Weight"]),
        }

    logger.info(f"Loaded FIFA rankings for {len(result)} teams")
    return result


def load_shootouts(path=None):
    """
    Load shootouts.csv for penalty shootout calibration.

    Returns
    -------
    pd.DataFrame
        Shootout results with normalized team names.
    """
    path = path or DATA_RAW / "shootouts.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    normalize_dataframe_teams(df, ["home_team", "away_team", "winner"])
    return df


def load_wc_historical_stats(path=None):
    """
    Load WC_historical_stats.csv (matches_1930_2022.csv from piterfm dataset).
    Contains WC match data with yellow/red cards, xG, attendance etc.

    Returns
    -------
    pd.DataFrame
        Historical WC match stats with normalized team names.
    """
    path = path or DATA_RAW / "WC_historical_stats.csv"
    try:
        df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
        normalize_dataframe_teams(df, ["home_team", "away_team"])
        logger.info(f"Loaded WC historical stats: {len(df)} matches ({df['Year'].min()}-{df['Year'].max()})")
        return df
    except Exception as e:
        logger.warning(f"Could not load WC_historical_stats.csv: {e}")
        return None


# ---------------------------------------------------------------------------
# TEAM REGISTRY BUILDER
# ---------------------------------------------------------------------------

def build_team_registry(group_fixtures_df, fifa_rankings_dict, save=True):
    """
    Build the master team registry with one row per WC 2026 team.

    Columns produced:
    - team_name, group, confederation, conf_weight
    - fifa_ranking, fifa_points
    - is_host (True for USA, Canada, Mexico)
    - is_placeholder (True for UEFA/FIFA playoff slots)

    Parameters
    ----------
    group_fixtures_df : pd.DataFrame
        Loaded group fixtures (from load_group_fixtures).
    fifa_rankings_dict : dict
        FIFA rankings dict (from load_fifa_rankings).
    save : bool
        If True, save as data/processed/team_registry.csv.

    Returns
    -------
    pd.DataFrame
        Team registry.
    """
    # All unique teams from fixtures
    home_teams = group_fixtures_df[["home_team", "group"]].rename(columns={"home_team": "team_name"})
    away_teams = group_fixtures_df[["away_team", "group"]].rename(columns={"away_team": "team_name"})
    all_teams = pd.concat([home_teams, away_teams]).drop_duplicates("team_name")

    rows = []
    for _, row in all_teams.iterrows():
        team = row["team_name"]
        group = row["group"]
        is_placeholder = "Playoff" in team or "playoff" in team

        # FIFA ranking data
        fi = fifa_rankings_dict.get(team, {})
        if not fi:
            # Try fallback with common name variants
            for alias, canon in NORMALIZATION_MAP.items():
                if canon == team and alias in fifa_rankings_dict:
                    fi = fifa_rankings_dict[alias]
                    break

        if not fi and not is_placeholder:
            logger.warning(f"No FIFA ranking data for confirmed team: {team}")

        # Confederation
        conf = fi.get("confederation") or CONFEDERATION_MAP.get(team, "Unknown")
        conf_w = fi.get("conf_weight") or CONF_WEIGHT.get(conf, 0.85)

        rows.append({
            "team_name":       team,
            "group":           group,
            "confederation":   conf,
            "conf_weight":     conf_w,
            "fifa_ranking":    fi.get("rank", 999 if not is_placeholder else None),
            "fifa_points":     fi.get("points", 1200 if is_placeholder else np.nan),
            "is_host":         team in HOST_NATIONS,
            "is_placeholder":  is_placeholder,
        })

    registry = pd.DataFrame(rows).sort_values(["group", "team_name"]).reset_index(drop=True)

    if save:
        out = DATA_PROCESSED / "team_registry.csv"
        registry.to_csv(out, index=False)
        logger.info(f"Saved team_registry.csv: {len(registry)} teams → {out}")

    return registry


# ---------------------------------------------------------------------------
# CONVENIENCE: LOAD ALL
# ---------------------------------------------------------------------------

def load_all(verbose=True):
    """
    Load all raw data sources and return a dict of DataFrames.

    Returns
    -------
    dict with keys:
        'results_full', 'results_primary', 'former_names',
        'group_fixtures', 'knockout_slots', 'fifa_rankings',
        'shootouts', 'wc_stats', 'team_registry'
    """
    results_full, results_primary = load_results()
    former_names = load_former_names()
    group_fixtures = load_group_fixtures()
    knockout_slots = load_knockout_slots()
    fifa_rankings = load_fifa_rankings()
    shootouts = load_shootouts()
    wc_stats = load_wc_historical_stats()
    team_registry = build_team_registry(group_fixtures, fifa_rankings, save=True)

    if verbose:
        print("\n" + "="*55)
        print("DATA LOADING COMPLETE")
        print("="*55)
        print(f"  results (full):       {len(results_full):>7,} rows")
        print(f"  results (1990+):      {len(results_primary):>7,} rows")
        print(f"  group fixtures:       {len(group_fixtures):>7,} matches")
        print(f"  knockout slots:       {len(knockout_slots):>7,} matches")
        print(f"  FIFA rankings:        {len(fifa_rankings):>7,} teams")
        print(f"  shootouts:            {len(shootouts):>7,} records")
        print(f"  WC historical stats:  {len(wc_stats) if wc_stats is not None else 'N/A':>7}")
        print(f"  team registry:        {len(team_registry):>7,} teams")
        print("="*55 + "\n")

    return {
        "results_full":     results_full,
        "results_primary":  results_primary,
        "former_names":     former_names,
        "group_fixtures":   group_fixtures,
        "knockout_slots":   knockout_slots,
        "fifa_rankings":    fifa_rankings,
        "shootouts":        shootouts,
        "wc_stats":         wc_stats,
        "team_registry":    team_registry,
    }
