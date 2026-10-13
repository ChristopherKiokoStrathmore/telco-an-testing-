"""Seeded generator for SYNTHETIC experiment data (not real operator data)."""
import numpy as np
import pandas as pd

REGIONS = ["Nairobi", "Coast", "Rift Valley", "Western", "Nyanza", "Central", "Eastern", "North Eastern"]


def generate(n=40_000, seed=42, base_cr=0.080, lift=0.012, arpu_effect=4.0):
    """Simulate a USSD prompt test: control = standard menu, treatment = personalised
    'Bundle ya Wiki' airtime offer prompt. Returns one row per subscriber."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "subscriber_id": np.arange(n),
        "region": rng.choice(REGIONS, n, p=[.28, .10, .18, .11, .10, .10, .10, .03]),
        "tenure_months": rng.gamma(2.0, 14.0, n).round(1),
        "group": rng.choice(["control", "treatment"], n),
    })
    # pre-period ARPU (KES / 30 days) -> used for CUPED
    df["pre_arpu"] = rng.lognormal(np.log(180), 0.6, n).round(2)
    t = (df["group"] == "treatment").astype(int)
    logit_base = np.log(base_cr / (1 - base_cr)) + 0.15 * np.log(df["pre_arpu"] / 180)
    p = 1 / (1 + np.exp(-logit_base))
    p = np.clip(p + t * lift, 0, 1)
    df["converted"] = rng.binomial(1, p)
    noise = rng.normal(0, 60, n)
    df["post_arpu"] = (0.85 * df["pre_arpu"] + 30 + t * arpu_effect
                       + df["converted"] * 25 + noise).clip(lower=0).round(2)
    # guardrails: opt-out from USSD prompts, complaint tickets
    df["opted_out"] = rng.binomial(1, 0.010 + t * 0.001)
    df["complaint"] = rng.binomial(1, 0.004 + t * 0.0003)
    return df
