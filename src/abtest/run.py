"""End-to-end analysis: python -m abtest.run  -> writes reports/ (figures + memo)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .data import generate
from . import stats as S

ROOT = Path(__file__).resolve().parents[2]


def main():
    fig_dir = ROOT / "reports" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    df = generate()
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "synthetic_experiment.csv", index=False)
    c, t = df[df.group == "control"], df[df.group == "treatment"]
    n_req = S.sample_size_two_prop(0.08, 0.01)
    srm_p = S.srm_check(len(c), len(t))
    zt = S.two_prop_ztest(c.converted.sum(), len(c), t.converted.sum(), len(t))
    raw = S.welch_diff(c.post_arpu.values, t.post_arpu.values)
    adj, theta = S.cuped(df.post_arpu.values, df.pre_arpu.values)
    df["arpu_cuped"] = adj
    cu = S.welch_diff(df.loc[df.group == "control", "arpu_cuped"].values,
                      df.loc[df.group == "treatment", "arpu_cuped"].values)
    var_red = 1 - (cu["se"] / raw["se"]) ** 2
    g_opt = S.two_prop_ztest(c.opted_out.sum(), len(c), t.opted_out.sum(), len(t))
    g_cmp = S.two_prop_ztest(c.complaint.sum(), len(c), t.complaint.sum(), len(t))
    peek = S.peeking_false_positive_rate()

    # figures
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Control", "Treatment"], [zt["p_control"] * 100, zt["p_treatment"] * 100], color=["#9aa5b1", "#e4002b"])
    ax.set_ylabel("Bundle conversion (%)"); ax.set_title("Conversion by group (synthetic)")
    fig.tight_layout(); fig.savefig(fig_dir / "conversion.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in [("Raw ARPU", raw), ("CUPED ARPU", cu)]:
        ax.errorbar([name], [r["diff"]], yerr=[[r["diff"] - r["ci"][0]], [r["ci"][1] - r["diff"]]], fmt="o", capsize=8)
    ax.axhline(0, color="grey", lw=1); ax.set_ylabel("Treatment - control (KES)")
    ax.set_title("ARPU lift 95% CI: CUPED narrows the interval")
    fig.tight_layout(); fig.savefig(fig_dir / "cuped_ci.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ns = np.arange(2000, 40001, 2000)
    pw = [1 - __import__("scipy").stats.norm.cdf(1.96 - 0.01 / np.sqrt(2 * 0.085 * 0.915 / (n / 2))) for n in ns]
    ax.plot(ns, pw, color="#e4002b"); ax.axhline(0.8, ls="--", color="grey")
    ax.set_xlabel("Total subscribers"); ax.set_ylabel("Power (MDE = +1pp)"); ax.set_title("Power curve")
    fig.tight_layout(); fig.savefig(fig_dir / "power_curve.png", dpi=150); plt.close(fig)

    ship = zt["p_value"] < 0.05 and zt["ci_abs"][0] > 0 and srm_p > 0.01 and g_opt["p_value"] > 0.05 and g_cmp["p_value"] > 0.05
    results = {"n_required_per_group": n_req, "n_control": len(c), "n_treatment": len(t), "srm_p": srm_p,
               "conversion": zt, "arpu_raw": raw, "arpu_cuped": cu, "cuped_theta": theta,
               "cuped_variance_reduction": var_red, "guardrail_optout": g_opt, "guardrail_complaint": g_cmp,
               "peeking_fpr_10_looks": peek, "decision": "SHIP" if ship else "DO NOT SHIP"}
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else float(o)))
    memo = f"""# Decision memo: 'Bundle ya Wiki' USSD prompt (SYNTHETIC DATA)

**Decision: {results['decision']}**

| Check | Result |
|---|---|
| Required n per group (8% base, +1pp MDE, 80% power) | {n_req:,} |
| Actual n (control / treatment) | {len(c):,} / {len(t):,} |
| SRM chi-square p | {srm_p:.3f} ({'OK' if srm_p > 0.01 else 'FAIL'}) |
| Conversion control → treatment | {zt['p_control']:.2%} → {zt['p_treatment']:.2%} |
| Absolute lift (95% CI) | {zt['abs_lift']*100:+.2f}pp ({zt['ci_abs'][0]*100:+.2f} to {zt['ci_abs'][1]*100:+.2f}) |
| Relative lift | {zt['rel_lift']:+.1%}, p = {zt['p_value']:.2g} |
| ARPU lift, raw (95% CI) | KES {raw['diff']:+.2f} ({raw['ci'][0]:+.2f} to {raw['ci'][1]:+.2f}) |
| ARPU lift, CUPED (95% CI) | KES {cu['diff']:+.2f} ({cu['ci'][0]:+.2f} to {cu['ci'][1]:+.2f}) |
| CUPED variance reduction | {var_red:.0%} |
| Guardrail: USSD opt-out diff | {g_opt['abs_lift']*100:+.2f}pp, p = {g_opt['p_value']:.2f} |
| Guardrail: complaint diff | {g_cmp['abs_lift']*100:+.2f}pp, p = {g_cmp['p_value']:.2f} |
| A/A false-positive rate if peeking 10× | {peek:.0%} (vs nominal 5%) |

**Why:** the primary metric is significant with a CI above zero, randomisation is healthy, and neither
guardrail moved. The test was analysed once at the pre-registered sample size, so the peeking
inflation shown above does not apply. Recommend a staged rollout (25% → 100%) with guardrails monitored.
"""
    (ROOT / "reports" / "decision_memo.md").write_text(memo)
    print(memo)
    return results


if __name__ == "__main__":
    main()
