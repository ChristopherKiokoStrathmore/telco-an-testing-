"""Core experiment statistics."""
import numpy as np
from scipy import stats


def sample_size_two_prop(p1, mde_abs, alpha=0.05, power=0.8):
    """Per-group n for a two-sided two-proportion test."""
    p2 = p1 + mde_abs
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / mde_abs ** 2))


def srm_check(n_control, n_treatment, expected=(0.5, 0.5)):
    """Chi-square sample-ratio-mismatch test. Returns p-value."""
    total = n_control + n_treatment
    exp = [total * expected[0], total * expected[1]]
    return stats.chisquare([n_control, n_treatment], exp).pvalue


def two_prop_ztest(x_c, n_c, x_t, n_t, alpha=0.05):
    pc, pt = x_c / n_c, x_t / n_t
    pool = (x_c + x_t) / (n_c + n_t)
    se0 = np.sqrt(pool * (1 - pool) * (1 / n_c + 1 / n_t))
    z = (pt - pc) / se0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se = np.sqrt(pc * (1 - pc) / n_c + pt * (1 - pt) / n_t)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = pt - pc
    return {"p_control": pc, "p_treatment": pt, "abs_lift": diff, "rel_lift": diff / pc,
            "z": z, "p_value": p, "ci_abs": (diff - zc * se, diff + zc * se)}


def welch_diff(y_c, y_t, alpha=0.05):
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(y_c.var(ddof=1) / len(y_c) + y_t.var(ddof=1) / len(y_t))
    zc = stats.norm.ppf(1 - alpha / 2)
    p = stats.ttest_ind(y_t, y_c, equal_var=False).pvalue
    return {"diff": diff, "se": se, "ci": (diff - zc * se, diff + zc * se), "p_value": p}


def cuped(y, x):
    """Return CUPED-adjusted metric and theta. y, x are arrays over all units."""
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), theta


def peeking_false_positive_rate(n_sims=500, n_per_group=5000, looks=10, p=0.08, seed=0):
    """Simulate A/A tests checked at `looks` interim points; stop at first p<0.05."""
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_group / looks, n_per_group, looks).astype(int)
    hits = 0
    for _ in range(n_sims):
        a, b = rng.binomial(1, p, n_per_group), rng.binomial(1, p, n_per_group)
        for k in checkpoints:
            if two_prop_ztest(a[:k].sum(), k, b[:k].sum(), k)["p_value"] < 0.05:
                hits += 1
                break
    return hits / n_sims
