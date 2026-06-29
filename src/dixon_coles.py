"""
src/dixon_coles.py

Phase 4.1: Dixon-Coles Bivariate Poisson Model for Score Prediction.

Fits attack (alpha) and defense (beta) parameters per team using
maximum-weighted log-likelihood. Includes:
- Low-scoring correction factor (rho)
- Time-decay weighting (xi)
- Home advantage parameter (gamma)
- Scoreline probability matrix P(i,j) for i,j in 0..7
- RPS validation on 2022 WC holdout

Reference:
  Dixon & Coles (1997) "Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market"
"""

import logging
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Maximum scoreline considered (P(i,j) matrix size)
MAX_GOALS = 8

# ---------------------------------------------------------------------------
# LOW-SCORE CORRECTION FACTOR
# ---------------------------------------------------------------------------

def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon-Coles low-score correction factor tau(x, y, lambda, mu, rho).

    Adjusts probabilities for 0-0, 1-0, 0-1, 1-1 outcomes.

    Parameters
    ----------
    x, y : int
        Home and away goals.
    lam, mu : float
        Expected goals (lambda_home, lambda_away).
    rho : float
        Dependency parameter (usually small, -0.1 to 0.1).

    Returns
    -------
    float
        Correction multiplier.
    """
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1.0 + mu * rho
    elif x == 0 and y == 1:
        return 1.0 + lam * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def score_probability(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    P(home=x, away=y) under Dixon-Coles bivariate Poisson model.

    Parameters
    ----------
    x : int  Home goals.
    y : int  Away goals.
    lam : float  E[home goals].
    mu : float   E[away goals].
    rho : float  Dependency parameter.

    Returns
    -------
    float  Probability of this exact scoreline.
    """
    return poisson.pmf(x, lam) * poisson.pmf(y, mu) * _tau(x, y, lam, mu, rho)


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    n = max_goals + 1
    idx = np.arange(n)
    # Fast manual Poisson PMF
    facts = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880][:n])
    
    pmf_lam = (lam ** idx) * np.exp(-lam) / facts
    pmf_mu = (mu ** idx) * np.exp(-mu) / facts
    
    mat = np.outer(pmf_lam, pmf_mu)
    
    # Apply tau corrections
    mat[0, 0] *= _tau(0, 0, lam, mu, rho)
    mat[1, 0] *= _tau(1, 0, lam, mu, rho)
    mat[0, 1] *= _tau(0, 1, lam, mu, rho)
    mat[1, 1] *= _tau(1, 1, lam, mu, rho)
    
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


def outcome_probs(mat: np.ndarray) -> tuple:
    """
    Derive P(home win), P(draw), P(away win) from a score matrix.

    Parameters
    ----------
    mat : np.ndarray  Score probability matrix.

    Returns
    -------
    (p_home, p_draw, p_away) tuple of floats.
    """
    n = mat.shape[0]
    p_home = sum(mat[i, j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(mat[i, i] for i in range(n))
    p_away = 1.0 - p_home - p_draw
    return float(p_home), float(p_draw), float(p_away)


def most_likely_scoreline(lam: float, mu: float, rho: float) -> tuple:
    """Return the (home_goals, away_goals) scoreline with highest probability."""
    mat = score_matrix(lam, mu, rho)
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    return int(idx[0]), int(idx[1])


# ---------------------------------------------------------------------------
# DIXON-COLES MODEL
# ---------------------------------------------------------------------------

class DixonColes:
    """
    Dixon-Coles bivariate Poisson model for predicting football scores.

    Parameters are fitted via maximum weighted log-likelihood with
    time-decay weighting (xi) and optional Bayesian regularisation for
    teams with few observations.

    Attributes
    ----------
    params_ : dict
        Fitted parameters: {team_alpha, team_beta, home_adv, rho, xi}
    teams_ : list
        All teams in the training set.
    """

    def __init__(self, xi: float = 0.003, rho: float = -0.1):
        """
        Parameters
        ----------
        xi : float
            Time-decay rate. Weight = exp(-xi * days_since_match).
            Higher xi → more weight on recent matches.
        rho : float
            Initial value of the dependency parameter rho (fitted by MLE).
        """
        self.xi = xi
        self.rho = rho
        self.params_ = None
        self.teams_ = None
        self._team_idx = None
        self._alpha = None   # attack parameters array
        self._beta = None    # defense parameters array
        self._home_adv = None
        self._rho = None

    # ── INTERNAL HELPERS ────────────────────────────────────────────────────

    def _negative_log_likelihood(self, params: np.ndarray,
                                  home_teams: np.ndarray, away_teams: np.ndarray,
                                  home_goals: np.ndarray, away_goals: np.ndarray,
                                  weights: np.ndarray) -> float:
        """
        Compute the weighted negative log-likelihood of the Dixon-Coles model.

        Parameters
        ----------
        params : np.ndarray
            Flattened parameter vector: [alpha_0..n, beta_0..n, home_adv, rho].
        home_teams, away_teams : np.ndarray[int]
            Index arrays into self.teams_.
        home_goals, away_goals : np.ndarray[int]
            Observed goals.
        weights : np.ndarray
            Match weights (time-decay).

        Returns
        -------
        float
            Negative weighted log-likelihood (to minimise).
        """
        n = len(self.teams_)
        alpha = params[:n]
        beta = params[n:2*n]
        home_adv = params[2*n]
        rho = params[2*n + 1]

        # Fully vectorized expected goals
        lam = np.exp(alpha[home_teams] + beta[away_teams] + home_adv)
        mu  = np.exp(alpha[away_teams] + beta[home_teams])

        # Vectorized tau (low-score correction)
        tau_val = np.ones_like(home_goals, dtype=float)
        
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_11 = (home_goals == 1) & (away_goals == 1)
        
        tau_val[mask_00] = 1.0 - lam[mask_00] * mu[mask_00] * rho
        tau_val[mask_10] = 1.0 + mu[mask_10] * rho
        tau_val[mask_01] = 1.0 + lam[mask_01] * rho
        tau_val[mask_11] = 1.0 - rho
        
        tau_val = np.maximum(tau_val, 1e-10)

        from scipy.special import gammaln
        # Fast vectorized poisson logpmf
        log_p_home = home_goals * np.log(lam) - lam - gammaln(home_goals + 1)
        log_p_away = away_goals * np.log(mu) - mu - gammaln(away_goals + 1)

        # Vectorized log probability
        log_p = (log_p_home + log_p_away + np.log(tau_val))

        # Weighted negative log-likelihood
        nll = -np.sum(weights * log_p)

        return nll

    def _compute_weights(self, dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
        """
        Compute time-decay weights for each match.

        weight = exp(-xi * days_since_match)

        Parameters
        ----------
        dates : pd.Series
        ref_date : pd.Timestamp
            Reference date (most recent match date).

        Returns
        -------
        np.ndarray
        """
        days = (ref_date - dates).dt.days.values.astype(float)
        return np.exp(-self.xi * days)

    # ── PUBLIC INTERFACE ────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp = None,
            min_matches: int = 5) -> "DixonColes":
        """
        Fit the Dixon-Coles model on historical match data.

        Parameters
        ----------
        df : pd.DataFrame
            Match data with columns: date, home_team, away_team,
            home_score, away_score. Must be pre-filtered (e.g. 2000+).
        ref_date : pd.Timestamp, optional
            Reference date for time-decay (defaults to max date in df).
        min_matches : int
            Teams with fewer matches get L2-regularised parameters.

        Returns
        -------
        self
        """
        df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"]).copy()
        df = df.sort_values("date").reset_index(drop=True)

        self.teams_ = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._team_idx = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)

        logger.info(f"Fitting Dixon-Coles on {len(df):,} matches, {n} teams")

        ref_date = ref_date or df["date"].max()
        weights = self._compute_weights(df["date"], ref_date)

        # Encode teams as indices
        home_idx = df["home_team"].map(self._team_idx).values
        away_idx = df["away_team"].map(self._team_idx).values
        hg = df["home_score"].values.astype(int)
        ag = df["away_score"].values.astype(int)

        # Count matches per team for regularisation
        match_counts = df["home_team"].value_counts().add(
            df["away_team"].value_counts(), fill_value=0
        )

        # Initial parameter vector: [alpha_0..n, beta_0..n, home_adv, rho]
        x0 = np.zeros(2 * n + 2)
        x0[2*n] = 0.2    # home advantage
        x0[2*n+1] = self.rho  # rho

        # Bounds: all params in [-3, 3], home_adv in [0, 0.5], rho in [-0.5, 0.5]
        bounds = [(-3, 3)] * (2 * n) + [(0, 0.5), (-0.5, 0.5)]

        logger.info("  Running L-BFGS-B optimisation...")
        result = minimize(
            self._negative_log_likelihood,
            x0,
            args=(home_idx, away_idx, hg, ag, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6},
        )

        if not result.success:
            logger.warning(f"  Optimisation did not fully converge: {result.message}")
        else:
            logger.info(f"  Converged in {result.nit} iterations. NLL={result.fun:.2f}")

        alpha = result.x[:n]
        beta  = result.x[n:2*n]
        self._home_adv = result.x[2*n]
        self._rho = result.x[2*n + 1]

        # Apply Bayesian shrinkage for data-sparse teams
        global_alpha = alpha.mean()
        global_beta  = beta.mean()
        for team, idx in self._team_idx.items():
            m = match_counts.get(team, 0)
            shrink = m / (m + min_matches)
            alpha[idx] = shrink * alpha[idx] + (1 - shrink) * global_alpha
            beta[idx]  = shrink * beta[idx]  + (1 - shrink) * global_beta

        self._alpha = alpha
        self._beta  = beta

        # Build readable params dict
        self.params_ = {
            "home_advantage": round(float(self._home_adv), 4),
            "rho":            round(float(self._rho), 4),
            "xi":             self.xi,
            "n_teams":        n,
            "teams":          {
                t: {"alpha": round(float(alpha[i]), 4),
                    "beta":  round(float(beta[i]), 4)}
                for t, i in self._team_idx.items()
            }
        }

        logger.info(f"  home_adv={self._home_adv:.4f}, rho={self._rho:.4f}")
        return self

    def _get_lambdas(self, home_team: str, away_team: str,
                     neutral: bool = True) -> tuple:
        """
        Compute expected goals (lambda_home, lambda_away) for a match.

        Parameters
        ----------
        home_team, away_team : str
        neutral : bool
            If True, apply reduced home advantage.

        Returns
        -------
        (float, float)  Expected goals for home and away teams.
        """
        if self._alpha is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        h_idx = self._team_idx.get(home_team)
        a_idx = self._team_idx.get(away_team)

        # Fallback to global mean for unseen teams
        alpha_h = self._alpha[h_idx] if h_idx is not None else self._alpha.mean()
        alpha_a = self._alpha[a_idx] if a_idx is not None else self._alpha.mean()
        beta_h  = self._beta[h_idx]  if h_idx is not None else self._beta.mean()
        beta_a  = self._beta[a_idx]  if a_idx is not None else self._beta.mean()

        if h_idx is None:
            pass # Suppressed unseen team warning
        if a_idx is None:
            pass # Suppressed unseen team warning

        # Apply partial home advantage at neutral venues (50% of fitted value)
        ha = self._home_adv * (0.5 if neutral else 1.0)

        lam = np.exp(alpha_h + beta_a + ha)
        mu  = np.exp(alpha_a + beta_h)

        return float(lam), float(mu)

    def predict_score_matrix(self, home_team: str, away_team: str,
                              neutral: bool = True) -> np.ndarray:
        """
        Predict full scoreline probability matrix for a match.

        Parameters
        ----------
        home_team, away_team : str
        neutral : bool

        Returns
        -------
        np.ndarray shape (MAX_GOALS+1, MAX_GOALS+1)
        """
        lam, mu = self._get_lambdas(home_team, away_team, neutral)
        return score_matrix(lam, mu, self._rho)

    def predict_outcome_probs(self, home_team: str, away_team: str,
                               neutral: bool = True) -> dict:
        """
        Predict P(home win), P(draw), P(away win).

        Parameters
        ----------
        home_team, away_team : str
        neutral : bool

        Returns
        -------
        dict with keys 'home', 'draw', 'away'.
        """
        mat = self.predict_score_matrix(home_team, away_team, neutral)
        ph, pd_, pa = outcome_probs(mat)
        lam, mu = self._get_lambdas(home_team, away_team, neutral)
        return {
            "home":     round(ph, 4),
            "draw":     round(pd_, 4),
            "away":     round(pa, 4),
            "exp_home_goals": round(lam, 3),
            "exp_away_goals": round(mu, 3),
        }

    def predict_scoreline(self, home_team: str, away_team: str,
                           neutral: bool = True) -> tuple:
        """
        Return the most likely predicted scoreline.

        Returns
        -------
        (int, int)  (home_goals, away_goals)
        """
        lam, mu = self._get_lambdas(home_team, away_team, neutral)
        return most_likely_scoreline(lam, mu, self._rho)

    def sample_scoreline(self, home_team: str, away_team: str,
                          neutral: bool = True, rng: np.random.Generator = None) -> tuple:
        """
        Sample a random scoreline from the Dixon-Coles distribution.
        Used in Monte Carlo tournament simulation.

        Parameters
        ----------
        home_team, away_team : str
        neutral : bool
        rng : np.random.Generator, optional

        Returns
        -------
        (int, int)  Sampled (home_goals, away_goals)
        """
        if rng is None:
            rng = np.random.default_rng()

        mat = self.predict_score_matrix(home_team, away_team, neutral)
        probs = mat.ravel()
        idx = rng.choice(len(probs), p=probs)
        h_idx, a_idx = divmod(idx, mat.shape[1])
        return int(h_idx), int(a_idx)

    # ── VALIDATION ──────────────────────────────────────────────────────────

    def validate(self, val_df: pd.DataFrame, neutral: bool = True) -> dict:
        """
        Validate on a holdout set (e.g. 2022 WC matches).

        Metrics: RPS, Brier score, accuracy (correct outcome).

        Parameters
        ----------
        val_df : pd.DataFrame
            Validation matches with home_team, away_team, home_score, away_score.
        neutral : bool
            Whether to treat all validation matches as neutral venue.

        Returns
        -------
        dict  {'rps': float, 'brier': float, 'accuracy': float, 'n': int}
        """
        rps_scores, brier_scores, correct = [], [], 0

        for _, row in val_df.iterrows():
            try:
                pred = self.predict_outcome_probs(
                    row["home_team"], row["away_team"], neutral
                )
                p_h, p_d, p_a = pred["home"], pred["draw"], pred["away"]
            except Exception:
                continue

            hg, ag = int(row["home_score"]), int(row["away_score"])
            if hg > ag:
                r_h, r_d, r_a = 1, 0, 0
            elif hg == ag:
                r_h, r_d, r_a = 0, 1, 0
            else:
                r_h, r_d, r_a = 0, 0, 1

            # Ranked Probability Score (RPS)
            cum_pred = np.cumsum([p_h, p_d, p_a])
            cum_real = np.cumsum([r_h, r_d, r_a])
            rps = 0.5 * np.sum((cum_pred[:2] - cum_real[:2]) ** 2)
            rps_scores.append(rps)

            # Brier score (sum of squared errors)
            brier = (p_h - r_h)**2 + (p_d - r_d)**2 + (p_a - r_a)**2
            brier_scores.append(brier)

            # Accuracy
            pred_outcome = np.argmax([p_h, p_d, p_a])
            actual_outcome = np.argmax([r_h, r_d, r_a])
            correct += int(pred_outcome == actual_outcome)

        n = len(rps_scores)
        return {
            "rps":      round(np.mean(rps_scores), 4) if rps_scores else None,
            "brier":    round(np.mean(brier_scores), 4) if brier_scores else None,
            "accuracy": round(correct / n, 4) if n > 0 else None,
            "n":        n,
        }

    def save_params(self, path=None):
        """Save model parameters to a CSV."""
        import json
        if path is None:
            path = Path(__file__).parent.parent / "outputs" / "model_artifacts" / "dixon_coles_params.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.params_, f, indent=2)
        logger.info(f"Saved Dixon-Coles params to {path}")


# ---------------------------------------------------------------------------
# GRID SEARCH FOR XI
# ---------------------------------------------------------------------------

def tune_xi(train_df: pd.DataFrame, val_df: pd.DataFrame,
             xi_values: list = None) -> dict:
    """
    Grid search for the best time-decay parameter xi.

    Parameters
    ----------
    train_df : pd.DataFrame  Training matches.
    val_df : pd.DataFrame    Validation matches (2022 WC).
    xi_values : list         Values of xi to try.

    Returns
    -------
    dict  {'best_xi': float, 'best_rps': float, 'results': list}
    """
    if xi_values is None:
        xi_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006]

    results = []
    logger.info(f"Grid searching xi over {xi_values}")

    for xi in xi_values:
        model = DixonColes(xi=xi)
        model.fit(train_df)
        metrics = model.validate(val_df, neutral=True)
        results.append({"xi": xi, **metrics})
        logger.info(f"  xi={xi:.3f} → RPS={metrics['rps']:.4f}, acc={metrics['accuracy']:.3f}")

    best = min(results, key=lambda x: x["rps"])
    logger.info(f"Best xi: {best['xi']:.3f} (RPS={best['rps']:.4f})")
    return {"best_xi": best["xi"], "best_rps": best["rps"], "results": results}
