"""
src/name_normalizer.py

Comprehensive team-name normalization for the FIFA WC 2026 prediction pipeline.

Handles:
 - Historical name changes (Zaire→Congo DR, West Germany→Germany, etc.)
 - Regional/dataset spelling variants (IR Iran, Korea Republic, etc.)
 - Character encoding variants (Türkiye, Côte d'Ivoire)
 - Dataset-specific quirks (results.csv vs group_fixtures.csv vs FIFA_Rankings.csv)
"""

# ---------------------------------------------------------------------------
# CANONICAL NORMALIZATION MAP
# Key   = name as it may appear in ANY source dataset
# Value = canonical name used throughout this pipeline
# ---------------------------------------------------------------------------
NORMALIZATION_MAP = {
    # ── United States ────────────────────────────────────────────────────────
    "United States":                        "USA",
    "United States of America":             "USA",
    "US":                                   "USA",
    "United States men's national soccer team": "USA",
    # ── Côte d'Ivoire ────────────────────────────────────────────────────────
    "Ivory Coast":                          "Côte d'Ivoire",
    "Ivory Coast national football team":   "Côte d'Ivoire",
    "Cote d'Ivoire":                        "Côte d'Ivoire",
    "Cote dIvoire":                         "Côte d'Ivoire",
    # ── Iran ─────────────────────────────────────────────────────────────────
    "IR Iran":                              "Iran",
    "Islamic Republic of Iran":             "Iran",
    # ── South Korea ──────────────────────────────────────────────────────────
    "Korea Republic":                       "South Korea",
    "Republic of Korea":                    "South Korea",
    # ── Bosnia and Herzegovina ───────────────────────────────────────────────
    "Bosnia-Herzegovina":                   "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":                 "Bosnia and Herzegovina",
    # ── Cabo Verde ───────────────────────────────────────────────────────────
    "Cape Verde":                           "Cabo Verde",
    "Cape Verde Islands":                   "Cabo Verde",
    "Cabo Verde national football team":    "Cabo Verde",
    # ── Congo DR ─────────────────────────────────────────────────────────────
    "Zaire":                                "Congo DR",
    "DR Congo":                             "Congo DR",
    "Democratic Republic of the Congo":     "Congo DR",
    "Congo, DR":                            "Congo DR",
    "Congo DR national football team":      "Congo DR",
    # ── Germany (historical) ─────────────────────────────────────────────────
    "West Germany":                         "Germany",
    "East Germany":                         "Germany",   # note: East Germany is a different team historically
    # ── Czechia ──────────────────────────────────────────────────────────────
    "Czech Republic":                       "Czechia",
    "Czechoslovakia":                       "Czechia",   # will cause issues pre-split; filter by date instead
    # ── Türkiye ──────────────────────────────────────────────────────────────
    "Turkey":                               "Türkiye",
    "Türkiye national football team":       "Türkiye",
    # ── Other common mismatches ───────────────────────────────────────────────
    "Republic of Ireland":                  "Ireland",
    "Northern Ireland":                     "Northern Ireland",  # keep as-is (different team)
    "China PR":                             "China",
    "Chinese Taipei":                       "Taiwan",
    "Chinese Taipei national football team": "Taiwan",
    "Trinidad and Tobago":                  "Trinidad & Tobago",
    "FYR Macedonia":                        "North Macedonia",
    "Macedonia":                            "North Macedonia",
    "Swaziland":                            "Eswatini",
    "São Tomé and Príncipe":               "Sao Tome and Principe",
    "Kyrgyzstan":                           "Kyrgyz Republic",
    "Korea DPR":                            "North Korea",
    "DPR Korea":                            "North Korea",
    # ── verbose FIFA_Rankings2-style names ───────────────────────────────────
    "Mexico national football team":              "Mexico",
    "South Africa national soccer team":          "South Africa",
    "Korea Republic national football team":      "South Korea",
    "Czechia national football team":             "Czechia",
    "Canada men's national soccer team":          "Canada",
    "Bosnia and Herzegovina national football team": "Bosnia and Herzegovina",
    "Qatar national football team":               "Qatar",
    "Switzerland national football team":         "Switzerland",
    "Brazil national football team":              "Brazil",
    "Morocco national football team":             "Morocco",
    "Haiti national football team":               "Haiti",
    "Scotland national football team":            "Scotland",
    "United States men's national soccer team":   "USA",
    "Australia national football team":           "Australia",
    "Paraguay national football team":            "Paraguay",
    "Germany national football team":             "Germany",
    "Ecuador national football team":             "Ecuador",
    "Netherlands national football team":         "Netherlands",
    "Japan national football team":               "Japan",
    "Sweden men's national football team":        "Sweden",
    "Tunisia national football team":             "Tunisia",
    "Belgium national football team":             "Belgium",
    "Egypt national football team":               "Egypt",
    "Iran national football team":                "Iran",
    "New Zealand national football team":         "New Zealand",
    "Spain national football team":               "Spain",
    "Uruguay national football team":             "Uruguay",
    "Saudi Arabia national football team":        "Saudi Arabia",
    "France national football team":              "France",
    "Senegal national football team":             "Senegal",
    "Iraq national football team":                "Iraq",
    "Norway national football team":              "Norway",
    "Argentina national football team":           "Argentina",
    "Austria national football team":             "Austria",
    "Algeria national football team":             "Algeria",
    "Jordan national football team":              "Jordan",
    "Portugal national football team":            "Portugal",
    "DR Congo national football team":            "Congo DR",
    "Uzbekistan national football team":          "Uzbekistan",
    "Colombia national football team":            "Colombia",
    "England national football team":             "England",
    "Croatia national football team":             "Croatia",
    "Ghana national football team":               "Ghana",
    "Panama national football team":              "Panama",
    "Curaçao national football team":             "Curaçao",
    "Ivory Coast national football team":         "Côte d'Ivoire",
}

# ---------------------------------------------------------------------------
# FIFA_Rankings.csv team names → canonical (only residual mismatches)
# ---------------------------------------------------------------------------
FIFA_RANKINGS_MAP = {
    "Czech Republic":   "Czechia",
    "Turkey":           "Türkiye",
    "Cape Verde":       "Cabo Verde",
    "DR Congo":         "Congo DR",
    "IR Iran":          "Iran",
    "Korea Republic":   "South Korea",
}

# ---------------------------------------------------------------------------
# Results.csv specific: IR Iran appears as-is; Korea Republic appears as-is
# These intentionally NOT mapped in NORMALIZATION_MAP so they propagate fine
# But in group_fixtures.csv they use the clean names — adjust accordingly
# ---------------------------------------------------------------------------
RESULTS_CSV_MAP = {
    "IR Iran":          "Iran",
    "Korea Republic":   "South Korea",
}


def normalize_team_name(name: str, extra_map: dict = None) -> str:
    """
    Normalize a team name to the pipeline's canonical form.

    Parameters
    ----------
    name : str
        Raw team name from any data source.
    extra_map : dict, optional
        Additional source-specific mappings to apply on top of NORMALIZATION_MAP.

    Returns
    -------
    str
        Canonical team name. Returns the input unchanged if no mapping found.
    """
    if not isinstance(name, str):
        return name

    name = name.strip()

    # Apply extra map first (highest priority, source-specific)
    if extra_map and name in extra_map:
        return extra_map[name]

    # Apply main normalization map
    return NORMALIZATION_MAP.get(name, name)


def normalize_dataframe_teams(df, columns, extra_map=None):
    """
    Apply normalize_team_name to specified columns of a DataFrame in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing team name columns.
    columns : list of str
        Column names to normalize (e.g. ['home_team', 'away_team']).
    extra_map : dict, optional
        Source-specific normalization map.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with team name columns normalized.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: normalize_team_name(x, extra_map))
    return df


# ---------------------------------------------------------------------------
# Historical name changes from former_names.csv
# Key = old name, Value = new canonical name, with optional date cutoff
# ---------------------------------------------------------------------------
FORMER_NAMES_SUPPLEMENT = {
    # These are loaded dynamically from former_names.csv, but listed here
    # for documentation and as a fallback if the file is missing.
    "Zaire":                "Congo DR",
    "West Germany":         "Germany",
    "Czechoslovakia":       "Czechia",     # rough approximation — technically split in 1993
    "Yugoslavia":           "Serbia",      # rough approximation
    "Soviet Union":         "Russia",      # rough approximation
    "Netherlands Antilles": "Curaçao",
    "Swaziland":            "Eswatini",
    "FYR Macedonia":        "North Macedonia",
    "Macedon":              "North Macedonia",
}


def build_former_names_map(former_names_path: str) -> dict:
    """
    Load the former_names.csv file and build a comprehensive name-change dict.

    Parameters
    ----------
    former_names_path : str
        Absolute path to former_names.csv.

    Returns
    -------
    dict
        Mapping of {old_name: current_name} for all historical name changes.
        Falls back to FORMER_NAMES_SUPPLEMENT if file is missing or unreadable.
    """
    import pandas as pd
    import logging
    logger = logging.getLogger(__name__)

    try:
        df = pd.read_csv(former_names_path)
        # Expected columns: 'former_name', 'current_name'  (or similar)
        # Inspect actual column names and adapt
        cols = df.columns.tolist()
        logger.info(f"former_names.csv columns: {cols}")

        # Try common column name patterns
        old_col = next((c for c in cols if 'former' in c.lower() or 'old' in c.lower() or 'previous' in c.lower()), None)
        new_col = next((c for c in cols if 'current' in c.lower() or 'new' in c.lower()), None)

        if old_col and new_col:
            mapping = dict(zip(df[old_col].str.strip(), df[new_col].str.strip()))
            # Merge with supplement (supplement is fallback for unmapped entries)
            for k, v in FORMER_NAMES_SUPPLEMENT.items():
                mapping.setdefault(k, v)
            logger.info(f"Loaded {len(mapping)} former name mappings")
            return mapping
        else:
            logger.warning(f"Could not identify columns in former_names.csv ({cols}). Using supplement.")
            return FORMER_NAMES_SUPPLEMENT.copy()

    except Exception as e:
        logger.warning(f"Failed to load former_names.csv: {e}. Using supplement.")
        return FORMER_NAMES_SUPPLEMENT.copy()
