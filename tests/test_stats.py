import numpy as np
from abtest import stats as S
from abtest.data import generate


def test_sample_size_reasonable():
    n = S.sample_size_two_prop(0.08, 0.01)
    assert 11000 < n < 13500


def test_srm_balanced_and_broken():
    assert S.srm_check(5000, 5010) > 0.5
    assert S.srm_check(5000, 5600) < 0.001


def test_ztest_detects_lift():
    df = generate(seed=1)
    c, t = df[df.group == "control"], df[df.group == "treatment"]
    r = S.two_prop_ztest(c.converted.sum(), len(c), t.converted.sum(), len(t))
    assert r["p_value"] < 0.05 and r["ci_abs"][0] < r["abs_lift"] < r["ci_abs"][1]


def test_cuped_reduces_variance():
    df = generate(seed=2)
    adj, theta = S.cuped(df.post_arpu.values, df.pre_arpu.values)
    assert theta > 0 and adj.var() < df.post_arpu.var()


def test_generator_is_deterministic():
    assert generate(n=500, seed=7).equals(generate(n=500, seed=7))


def test_peeking_inflates_fpr():
    assert S.peeking_false_positive_rate(n_sims=200, n_per_group=2000) > 0.07
