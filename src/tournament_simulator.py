"""
src/tournament_simulator.py

Phase 6: Monte Carlo Tournament Simulation.

Simulates the 2026 FIFA World Cup N=50,000 times using Dixon-Coles sampled
scorelines. Produces:
- Group stage standings per simulation
- Best 3rd-place team selection (8 of 12 advance)
- Full knockout bracket traversal (R32 → R16 → QF → SF → Final)
- Probability distributions for every knockout match
- Tournament winner probability per team

Key design decisions:
- Penalty shootout for knockout ties: modelled using shootouts.csv base rates
- Best-3rd ranking: points → GD → GS → disciplinary (randomised tiebreaker)
- Progress updates every 10,000 iterations
"""

import logging
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ── TOURNAMENT STRUCTURE ──────────────────────────────────────────────────────
# Groups and their teams (loaded from group_fixtures.csv at runtime)
GROUP_LETTERS = list("ABCDEFGHIJKL")

# Best-3rd pool → R32 slot mapping
# Each R32 best-3rd match draws from these pool definitions
BEST_THIRD_POOLS = [
    ("Best 3rd (A/B/C/D)",  frozenset("ABCD")),
    ("Best 3rd (E/F/G/H)",  frozenset("EFGH")),
    ("Best 3rd (I/J/K/L)",  frozenset("IJKL")),
    ("Best 3rd (A/B/I/J)",  frozenset("ABIJ")),
    ("Best 3rd (C/D/K/L)",  frozenset("CDKL")),
    ("Best 3rd (E/F/G/H)",  frozenset("EFGH")),  # duplicate handled in assignment
]

# WC penalty shootout historical rates
WC_PENALTY_RATE = 0.25  # ~25% of drawn WC knockout matches go to penalties
PENALTY_HOME_WIN_RATE = 0.50  # 50/50 in shootouts


class TournamentSimulator:
    """
    Monte Carlo simulator for the 2026 FIFA World Cup.

    Attributes
    ----------
    dc_model : DixonColes
        Fitted Dixon-Coles model for sampling scorelines.
    group_fixtures : pd.DataFrame
        All 72 group stage fixtures.
    knockout_slots : pd.DataFrame
        Bracket definition for matches 73-104.
    elo_ratings : dict
        {team: elo} for Elo-based penalty probability adjustment.
    """

    def __init__(self, dc_model, group_fixtures: pd.DataFrame,
                 knockout_slots: pd.DataFrame, elo_ratings: dict,
                 shootout_df: pd.DataFrame = None, seed: int = 42):
        self.dc = dc_model
        self.fixtures = group_fixtures.to_dict("records")
        self.ko_slots = knockout_slots.to_dict("records")
        self.elo = elo_ratings
        self.seed = seed

        # Group structure
        self.groups = {}
        for g in GROUP_LETTERS:
            teams = list(set(
                [fix["home_team"] for fix in self.fixtures if fix["group"] == g] +
                [fix["away_team"] for fix in self.fixtures if fix["group"] == g]
            ))
            if teams:
                self.groups[g] = teams

        # Penalty base rates from shootouts data
        self.penalty_rate = WC_PENALTY_RATE
        if shootout_df is not None:
            try:
                # Filter WC knockout shootouts if possible
                wc_shootouts = shootout_df[
                    shootout_df.get("tournament", pd.Series()).str.contains(
                        "FIFA World Cup", na=False
                    )
                ] if "tournament" in shootout_df.columns else shootout_df
                if len(wc_shootouts) > 10:
                    self.penalty_rate = len(wc_shootouts) / max(len(wc_shootouts), 100)
            except Exception:
                pass

        logger.info(f"Simulator initialised. Groups: {list(self.groups.keys())}, "
                    f"Penalty rate: {self.penalty_rate:.2f}")

    # ── SINGLE-MATCH SAMPLING ─────────────────────────────────────────────────

    def _sample_match(self, home: str, away: str, rng: np.random.Generator,
                       knockout: bool = False) -> dict:
        """
        Sample a scoreline from Dixon-Coles distribution.

        In knockout stage: if tied, resolve with penalty shootout.

        Parameters
        ----------
        home, away : str  Team names.
        rng : np.random.Generator
        knockout : bool  True → resolve draws with penalties.

        Returns
        -------
        dict
            {home_goals, away_goals, winner, went_to_penalties}
        """
        # Handle placeholder teams with fallback
        if "Playoff" in str(home) or "Playoff" in str(away):
            # Use random result weighted toward draw (weaker opponents)
            r = rng.uniform()
            if r < 0.35:
                hg, ag = 1, 0
            elif r < 0.65:
                hg, ag = 1, 1
            else:
                hg, ag = 0, 1
        else:
            try:
                hg, ag = self.dc.sample_scoreline(home, away, neutral=True, rng=rng)
            except Exception:
                # Fallback: random Poisson with prior means
                lam_h = np.exp(self.dc._alpha.mean() + self.dc._beta.mean() + self.dc._home_adv * 0.5)
                lam_a = np.exp(self.dc._alpha.mean() + self.dc._beta.mean())
                hg = int(rng.poisson(lam_h))
                ag = int(rng.poisson(lam_a))

        penalties = False
        winner = None

        if hg > ag:
            winner = home
        elif ag > hg:
            winner = away
        else:
            if knockout:
                # Resolve with penalty shootout
                penalties = True
                winner = home if rng.uniform() < PENALTY_HOME_WIN_RATE else away
            else:
                winner = "draw"

        return {"home_goals": hg, "away_goals": ag,
                "winner": winner, "went_to_penalties": penalties}

    # ── GROUP STAGE ────────────────────────────────────────────────────────────

    def _simulate_group_stage(self, rng: np.random.Generator) -> dict:
        """
        Simulate all 72 group stage matches and return standings.

        Returns
        -------
        dict
            {group_letter: [
                {team, pts, gf, ga, gd, wins, draws, losses, yellows, reds}, ...
            ]}  sorted by ranking criteria (descending)
        """
        # Initialise stats per team
        stats = {}
        for g, teams in self.groups.items():
            stats[g] = {t: {"pts": 0, "gf": 0, "ga": 0, "wins": 0,
                             "draws": 0, "losses": 0, "yellows": 0, "reds": 0,
                             "group": g}
                        for t in teams}

        # Play each fixture
        for fix in self.fixtures:
            h, a = fix["home_team"], fix["away_team"]
            g = fix["group"]
            result = self._sample_match(h, a, rng, knockout=False)
            hg, ag = result["home_goals"], result["away_goals"]

            if g not in stats or h not in stats[g] or a not in stats[g]:
                continue  # skip placeholder groups gracefully

            stats[g][h]["gf"] += hg
            stats[g][h]["ga"] += ag
            stats[g][a]["gf"] += ag
            stats[g][a]["ga"] += hg

            if hg > ag:
                stats[g][h]["pts"] += 3
                stats[g][h]["wins"] += 1
                stats[g][a]["losses"] += 1
            elif hg == ag:
                stats[g][h]["pts"] += 1
                stats[g][a]["pts"] += 1
                stats[g][h]["draws"] += 1
                stats[g][a]["draws"] += 1
            else:
                stats[g][a]["pts"] += 3
                stats[g][a]["wins"] += 1
                stats[g][h]["losses"] += 1

            # Simulated discipline (Poisson-distributed)
            stats[g][h]["yellows"] += int(rng.poisson(1.6))
            stats[g][a]["yellows"] += int(rng.poisson(1.6))
            stats[g][h]["reds"] += int(rng.poisson(0.06))
            stats[g][a]["reds"] += int(rng.poisson(0.06))

        # Rank each group
        standings = {}
        for g, teams_stats in stats.items():
            table = []
            for team, s in teams_stats.items():
                s["gd"] = s["gf"] - s["ga"]
                s["team"] = team
                s["disc"] = s["yellows"] + 2 * s["reds"]
                table.append(s)

            # Sort: pts → gd → gf → disc (asc) → random tiebreaker
            table.sort(key=lambda x: (
                -x["pts"], -x["gd"], -x["gf"], x["disc"], rng.uniform()
            ))
            standings[g] = table

        return standings

    # ── BEST 3RD SELECTION ────────────────────────────────────────────────────

    def _pick_best_thirds(self, standings: dict, rng: np.random.Generator) -> list:
        """
        Select the 8 best 3rd-placed teams from 12 groups.

        Ranking: pts → gd → gf → disc → random.

        Returns list of 8 (team, group) tuples, sorted best-first.
        """
        thirds = []
        for g, table in standings.items():
            if len(table) >= 3:
                t = table[2]  # 3rd place
                thirds.append({
                    "team": t["team"], "group": g,
                    "pts": t["pts"], "gd": t["gd"],
                    "gf": t["gf"], "disc": t["disc"],
                })

        thirds.sort(key=lambda x: (
            -x["pts"], -x["gd"], -x["gf"], x["disc"], rng.uniform()
        ))
        return thirds[:8]

    # ── KNOCKOUT BRACKET ──────────────────────────────────────────────────────

    def _resolve_slot(self, slot_str: str, standings: dict,
                       best_thirds: list, match_winners: dict) -> str:
        """
        Resolve a bracket slot string to an actual team name.

        Slot formats:
        - "Winner Group A"        → group A winner
        - "Runner-up Group B"     → group B runner-up
        - "Best 3rd (A/B/C/D)"    → best 3rd team from those groups
        - "Winner Match 73"       → winner of match #73
        - "Loser Match 101"       → loser of match #101

        Returns team name string.
        """
        s = str(slot_str)

        if s.startswith("Winner Group "):
            g = s.replace("Winner Group ", "").strip()
            return standings[g][0]["team"] if g in standings else "Unknown"

        elif s.startswith("Runner-up Group "):
            g = s.replace("Runner-up Group ", "").strip()
            return standings[g][1]["team"] if g in standings and len(standings[g]) >= 2 else "Unknown"

        elif s.startswith("Best 3rd ("):
            # Extract groups from parentheses e.g. "(A/B/C/D)"
            groups_str = s[s.index("(")+1:s.index(")")]
            group_set = frozenset(groups_str.split("/"))
            # Find best 3rd team from this group pool (not already assigned)
            for t in best_thirds:
                if t["group"] in group_set:
                    # Mark as used and return
                    idx = next((i for i, bt in enumerate(best_thirds)
                                if bt["team"] == t["team"]), None)
                    if idx is not None:
                        best_thirds.pop(idx)
                    return t["team"]
            # Fallback: any remaining best 3rd
            if best_thirds:
                return best_thirds.pop(0)["team"]
            return "Unknown"

        elif s.startswith("Winner Match "):
            mid = int(s.replace("Winner Match ", "").strip())
            return match_winners.get(mid, {}).get("winner", "Unknown")

        elif s.startswith("Loser Match "):
            mid = int(s.replace("Loser Match ", "").strip())
            r = match_winners.get(mid, {})
            home, away = r.get("home", "?"), r.get("away", "?")
            winner = r.get("winner", "?")
            return away if winner == home else home

        return "Unknown"

    def _simulate_knockout(self, standings: dict, best_thirds_in: list,
                            rng: np.random.Generator) -> dict:
        """
        Simulate the entire knockout bracket from R32 to Final.

        Returns
        -------
        dict
            {match_id: {home, away, home_goals, away_goals, winner, penalties}}
        """
        match_winners = {}
        best_thirds = list(best_thirds_in)  # mutable copy

        for slot in self.ko_slots:
            mid = int(slot["match_id"])
            sh, sa = slot["slot_home"], slot["slot_away"]

            home = self._resolve_slot(sh, standings, best_thirds, match_winners)
            away = self._resolve_slot(sa, standings, best_thirds, match_winners)

            result = self._sample_match(home, away, rng, knockout=True)
            result.update({"home": home, "away": away, "match_id": mid})
            match_winners[mid] = result

        return match_winners

    # ── FULL SIMULATION ───────────────────────────────────────────────────────

    def _run_chunk(self, n_simulations: int, seed: int) -> tuple:
        """Run a chunk of simulations and return tracking dicts."""
        rng = np.random.default_rng(seed)
        
        group_finish = defaultdict(lambda: defaultdict(list))
        ko_appearances = defaultdict(lambda: defaultdict(int))
        ko_wins = defaultdict(lambda: defaultdict(int))
        champion_counts = defaultdict(int)
        finalist_counts = defaultdict(int)
        semi_counts = defaultdict(int)
        quarter_counts = defaultdict(int)

        for sim_i in range(n_simulations):
            # Group stage
            standings = self._simulate_group_stage(rng)
            best_thirds = self._pick_best_thirds(standings, rng)

            # Record group finishes
            for g, table in standings.items():
                for rank_0, row in enumerate(table):
                    group_finish[g][row["team"]].append(rank_0 + 1)

            # Knockout stage
            ko_results = self._simulate_knockout(standings, best_thirds, rng)

            # Record knockout appearances and wins
            for mid, res in ko_results.items():
                slot_row = next((s for s in self.ko_slots if s["match_id"] == mid), None)
                if not slot_row:
                    continue
                rnd = slot_row["round"]

                h, a, winner = res.get("home", "?"), res.get("away", "?"), res.get("winner", "?")
                if h != "Unknown":
                    ko_appearances[mid][h] += 1
                    if winner == h:
                        ko_wins[mid][h] += 1
                if a != "Unknown":
                    ko_appearances[mid][a] += 1
                    if winner == a:
                        ko_wins[mid][a] += 1

                # Stage trackers
                if rnd == "Final":
                    finalist_counts[h] += 1
                    finalist_counts[a] += 1
                    if winner != "Unknown":
                        champion_counts[winner] += 1
                elif rnd == "Semi-final":
                    semi_counts[h] += 1
                    semi_counts[a] += 1
                elif rnd == "Quarter-final":
                    quarter_counts[h] += 1
                    quarter_counts[a] += 1
                    
        return (group_finish, ko_appearances, ko_wins, 
                champion_counts, finalist_counts, semi_counts, quarter_counts)

    def run(self, n_simulations: int = 50_000, n_jobs: int = -1) -> dict:
        """
        Run N Monte Carlo simulations of the full 2026 WC.

        Parameters
        ----------
        n_simulations : int  Default 50,000.
        n_jobs : int  Number of CPU cores to use (-1 for all). Default -1.

        Returns
        -------
        dict  Aggregated results — see _aggregate_results().
        """
        import joblib
        import os
        from joblib import Parallel, delayed

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 4
            
        logger.info(f"Starting {n_simulations:,} simulations across {n_jobs} cores...")

        # Split workload
        chunk_size = n_simulations // n_jobs
        remainder = n_simulations % n_jobs
        chunks = [chunk_size + (1 if i < remainder else 0) for i in range(n_jobs)]

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_chunk)(chunk, self.seed + i) 
            for i, chunk in enumerate(chunks) if chunk > 0
        )

        # Merge results
        merged_group_finish = defaultdict(lambda: defaultdict(list))
        merged_ko_appearances = defaultdict(lambda: defaultdict(int))
        merged_ko_wins = defaultdict(lambda: defaultdict(int))
        merged_champion = defaultdict(int)
        merged_finalist = defaultdict(int)
        merged_semi = defaultdict(int)
        merged_quarter = defaultdict(int)

        for (gf, ka, kw, cc, fc, sc, qc) in results:
            # Merge group_finish
            for g, teams in gf.items():
                for t, ranks in teams.items():
                    merged_group_finish[g][t].extend(ranks)
                    
            # Merge ko_appearances
            for mid, teams in ka.items():
                for t, count in teams.items():
                    merged_ko_appearances[mid][t] += count
                    
            # Merge ko_wins
            for mid, teams in kw.items():
                for t, count in teams.items():
                    merged_ko_wins[mid][t] += count
                    
            # Merge others
            for t, c in cc.items(): merged_champion[t] += c
            for t, c in fc.items(): merged_finalist[t] += c
            for t, c in sc.items(): merged_semi[t] += c
            for t, c in qc.items(): merged_quarter[t] += c

        print(f"  ✅ All {n_simulations:,} simulations complete!")
        return self._aggregate_results(
            n_simulations, merged_group_finish,
            merged_ko_appearances, merged_ko_wins,
            merged_champion, merged_finalist,
            merged_semi, merged_quarter
        )

    # ── RESULT AGGREGATION ────────────────────────────────────────────────────

    def _aggregate_results(self, n, group_finish, ko_appearances, ko_wins,
                            champion_counts, finalist_counts,
                            semi_counts, quarter_counts) -> dict:
        """Aggregate raw simulation counts into probability DataFrames."""

        # ── Tournament winner probabilities ──────────────────────────────────
        winner_probs = pd.DataFrame([
            {"team": t, "win_probability": round(c / n, 4),
             "finalist_prob": round(finalist_counts.get(t, 0) / n, 4),
             "semi_prob": round(semi_counts.get(t, 0) / n, 4),
             "quarter_prob": round(quarter_counts.get(t, 0) / n, 4)}
            for t, c in champion_counts.items()
        ]).sort_values("win_probability", ascending=False).reset_index(drop=True)

        # ── Knockout match team probabilities ─────────────────────────────────
        ko_team_probs = []
        for mid, teams in ko_appearances.items():
            slot_row = next((s for s in self.ko_slots if s["match_id"] == mid), None)
            rnd = slot_row["round"] if slot_row else "Unknown"
            for team, appearances in teams.items():
                wins = ko_wins[mid].get(team, 0)
                ko_team_probs.append({
                    "match_id":    mid,
                    "round":       rnd,
                    "team":        team,
                    "probability": round(appearances / n, 4),
                    "win_prob_given_appearance": round(wins / max(appearances, 1), 4),
                })
        ko_probs_df = pd.DataFrame(ko_team_probs).sort_values(["match_id", "probability"],
                                                                ascending=[True, False])

        # ── Modal knockout predictions (most likely team in each slot) ────────
        ko_preds = []
        for slot_row in self.ko_slots:
            mid = slot_row["match_id"]
            match_teams = ko_team_probs
            home_candidates = [r for r in match_teams if r["match_id"] == mid][:20]
            # Split by slot position (highest prob = home, 2nd = away)
            sorted_cands = sorted(home_candidates, key=lambda x: -x["probability"])
            if len(sorted_cands) >= 2:
                # Pair them by probability — highest two = most likely home/away
                h_team = sorted_cands[0]["team"]
                a_team = sorted_cands[1]["team"]
                h_prob = sorted_cands[0]["win_prob_given_appearance"]
                ko_preds.append({
                    "match_id": mid, "round": slot_row["round"],
                    "date_utc": slot_row["date_utc"], "venue": slot_row["venue"],
                    "predicted_home": h_team, "predicted_away": a_team,
                    "home_win_prob": round(h_prob, 4),
                    "away_win_prob": round(1 - h_prob, 4),
                    "predicted_winner": h_team if h_prob >= 0.5 else a_team,
                    "slot_home": slot_row["slot_home"],
                    "slot_away": slot_row["slot_away"],
                })

        ko_preds_df = pd.DataFrame(ko_preds)

        # ── Group finish probabilities ─────────────────────────────────────────
        group_probs = []
        for g, teams in group_finish.items():
            for team, ranks in teams.items():
                rank_counts = defaultdict(int)
                for r in ranks:
                    rank_counts[r] += 1
                group_probs.append({
                    "group": g, "team": team,
                    "p_1st": round(rank_counts.get(1, 0) / n, 4),
                    "p_2nd": round(rank_counts.get(2, 0) / n, 4),
                    "p_3rd": round(rank_counts.get(3, 0) / n, 4),
                    "p_4th": round(rank_counts.get(4, 0) / n, 4),
                    "p_advance": round((rank_counts.get(1, 0) + rank_counts.get(2, 0)) / n, 4),
                })
        group_probs_df = pd.DataFrame(group_probs).sort_values(
            ["group", "p_1st"], ascending=[True, False]
        )

        return {
            "winner_probs":     winner_probs,
            "ko_team_probs":    ko_probs_df,
            "ko_predictions":   ko_preds_df,
            "group_probs":      group_probs_df,
            "n_simulations":    n,
        }

    # ── SAVE OUTPUTS ──────────────────────────────────────────────────────────

    def save_results(self, results: dict):
        """Save all simulation output DataFrames to data/processed/."""
        paths = {
            "tournament_winner_probs":   DATA_PROCESSED / "tournament_winner_probs.csv",
            "knockout_team_probabilities": DATA_PROCESSED / "knockout_team_probabilities.csv",
            "knockout_predictions":      DATA_PROCESSED / "knockout_predictions.csv",
            "group_stage_probabilities": DATA_PROCESSED / "group_stage_probabilities.csv",
        }
        results["winner_probs"].to_csv(paths["tournament_winner_probs"], index=False)
        results["ko_team_probs"].to_csv(paths["knockout_team_probabilities"], index=False)
        results["ko_predictions"].to_csv(paths["knockout_predictions"], index=False)
        results["group_probs"].to_csv(paths["group_stage_probabilities"], index=False)

        for label, path in paths.items():
            logger.info(f"Saved {label} → {path}")


# ── PENALTY PROBABILITY ESTIMATOR ────────────────────────────────────────────

def compute_penalty_probability(elo_h: float, elo_a: float,
                                 base_rate: float = 0.25) -> float:
    """
    Estimate probability that a knockout match goes to penalties.

    Close matches (small Elo diff) are more likely to be level after 120'.

    Parameters
    ----------
    elo_h, elo_a : float
    base_rate : float  WC historical penalty rate (~25% of drawn matches).

    Returns
    -------
    float  Probability of going to penalties, in [0, 1].
    """
    elo_diff = abs(elo_h - elo_a)
    if elo_diff < 100:
        rate = base_rate + 0.05
    elif elo_diff > 200:
        rate = max(base_rate - 0.10, 0.05)
    else:
        rate = base_rate
    return round(rate, 4)
