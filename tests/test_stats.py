"""Unit tests for scanner.stats — Wilson interval + coin-flip verdict.

Reference CIs are the standard Wilson 95% score intervals (verifiable against
any stats table / statsmodels proportion_confint(method='wilson')). Run with:
    .venv/bin/python -m unittest tests.test_stats -v
"""
from __future__ import annotations

import unittest

from scanner import stats


class TestWilsonInterval(unittest.TestCase):
    def test_fifty_fifty_straddles_half(self):
        lo, hi = stats.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.404, places=2)
        self.assertAlmostEqual(hi, 0.596, places=2)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_strong_edge_entirely_above_half(self):
        lo, hi = stats.wilson_interval(90, 100)
        self.assertGreater(lo, 0.5)
        self.assertAlmostEqual(lo, 0.825, places=2)
        self.assertAlmostEqual(hi, 0.945, places=2)

    def test_bounds_clamped_to_unit_interval(self):
        lo, hi = stats.wilson_interval(20, 20)  # p = 1.0
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLessEqual(lo, hi)

    def test_zero_n_is_none(self):
        self.assertIsNone(stats.wilson_interval(0, 0))

    def test_wider_z_gives_wider_interval(self):
        lo95, hi95 = stats.wilson_interval(30, 50)
        lo99, hi99 = stats.wilson_interval(30, 50, z=2.5758)
        self.assertLess(lo99, lo95)
        self.assertGreater(hi99, hi95)


class TestCoinFlipVerdict(unittest.TestCase):
    def test_coin_flip_when_straddling(self):
        self.assertEqual(stats.coin_flip_verdict(52, 100), "coin_flip")

    def test_edge_when_clearly_above(self):
        self.assertEqual(stats.coin_flip_verdict(90, 100), "edge")

    def test_adverse_when_clearly_below(self):
        self.assertEqual(stats.coin_flip_verdict(10, 100), "adverse")

    def test_no_data_when_empty(self):
        self.assertEqual(stats.coin_flip_verdict(0, 0), "no_data")

    def test_small_n_high_rate_still_coin_flip(self):
        # 4/5 = 80% looks great but n is tiny → CI is wide → not distinguishable.
        self.assertEqual(stats.coin_flip_verdict(4, 5), "coin_flip")


class TestBinomSummary(unittest.TestCase):
    def test_shape_and_values(self):
        s = stats.binom_summary(54, 100)
        self.assertEqual(s["n"], 100)
        self.assertEqual(s["hits"], 54)
        self.assertEqual(s["hit_rate"], 0.54)
        self.assertIsNotNone(s["ci_low"])
        self.assertIsNotNone(s["ci_high"])
        self.assertEqual(s["verdict"], "coin_flip")

    def test_empty_bucket(self):
        s = stats.binom_summary(0, 0)
        self.assertIsNone(s["hit_rate"])
        self.assertIsNone(s["ci_low"])
        self.assertEqual(s["verdict"], "no_data")


class TestBlockBootstrap(unittest.TestCase):
    def _blocks(self, per_day: list[list[float]]):
        return [(f"d{i}", vals) for i, vals in enumerate(per_day)]

    def test_clear_positive_edge_excludes_zero(self):
        b = self._blocks([[1.0, 1.1, 0.9]] * 40)  # every day strongly positive
        r = stats.block_bootstrap_mean_ci(b, n_boot=500, seed=1)
        self.assertGreater(r["ci_low"], 0)
        self.assertEqual(stats.zero_crossing_verdict(r["ci_low"], r["ci_high"]), "edge")
        self.assertLess(r["p_value"], 0.05)

    def test_symmetric_noise_straddles_zero(self):
        # Days alternate + and - around 0 → mean ~0, CI straddles.
        per_day = [[0.5, 0.4] if i % 2 else [-0.5, -0.4] for i in range(40)]
        r = stats.block_bootstrap_mean_ci(self._blocks(per_day), n_boot=500, seed=2)
        self.assertLessEqual(r["ci_low"], 0)
        self.assertGreaterEqual(r["ci_high"], 0)
        self.assertEqual(stats.zero_crossing_verdict(r["ci_low"], r["ci_high"]), "coin_flip")

    def test_effective_n_is_days_not_observations(self):
        b = self._blocks([[1.0, 2.0, 3.0]] * 10)  # 10 days, 30 observations
        r = stats.block_bootstrap_mean_ci(b, n_boot=100, seed=3)
        self.assertEqual(r["n_obs"], 30)
        self.assertEqual(r["n_eff"], 10)   # honest sample size = days, not ticker-days

    def test_deterministic_under_seed(self):
        b = self._blocks([[0.3, -0.1, 0.2]] * 25)
        r1 = stats.block_bootstrap_mean_ci(b, n_boot=300, seed=7)
        r2 = stats.block_bootstrap_mean_ci(b, n_boot=300, seed=7)
        self.assertEqual(r1["ci_low"], r2["ci_low"])
        self.assertEqual(r1["ci_high"], r2["ci_high"])

    def test_empty_is_none(self):
        r = stats.block_bootstrap_mean_ci([], n_boot=100, seed=1)
        self.assertIsNone(r["ci_low"])
        self.assertEqual(r["n_obs"], 0)


class TestBlockBootstrapDiff(unittest.TestCase):
    def _blocks(self, per_day: list[list[float]]):
        return [(f"d{i}", vals) for i, vals in enumerate(per_day)]

    def test_clear_positive_diff_excludes_zero(self):
        a = self._blocks([[1.0, 1.1, 0.9]] * 40)   # cohort A ~ +1.0 every day
        b = self._blocks([[0.0, 0.1, -0.1]] * 40)  # cohort B ~ 0 every day
        r = stats.block_bootstrap_diff_ci(a, b, n_boot=500, seed=1)
        self.assertGreater(r["edge"], 0.8)
        self.assertGreater(r["ci_low"], 0)
        self.assertEqual(stats.zero_crossing_verdict(r["ci_low"], r["ci_high"]), "edge")
        self.assertLess(r["p_value"], 0.05)

    def test_identical_cohorts_straddle_zero(self):
        same = [[0.5, -0.4, 0.2]] * 40
        r = stats.block_bootstrap_diff_ci(self._blocks(same), self._blocks(same),
                                          n_boot=500, seed=2)
        self.assertLessEqual(r["ci_low"], 0)
        self.assertGreaterEqual(r["ci_high"], 0)
        self.assertEqual(stats.zero_crossing_verdict(r["ci_low"], r["ci_high"]), "coin_flip")

    def test_effective_sample_is_union_of_dates(self):
        a = [("d0", [1.0]), ("d1", [1.0])]
        b = [("d1", [0.0]), ("d2", [0.0])]  # overlaps on d1 only
        r = stats.block_bootstrap_diff_ci(a, b, n_boot=100, seed=3)
        self.assertEqual(r["n_days"], 3)  # d0,d1,d2
        self.assertEqual(r["n_a"], 2)
        self.assertEqual(r["n_b"], 2)

    def test_empty_side_is_none(self):
        r = stats.block_bootstrap_diff_ci([("d0", [1.0])], [], n_boot=100, seed=1)
        self.assertIsNone(r["edge"])
        self.assertEqual(r["n_b"], 0)

    def test_deterministic_under_seed(self):
        a = self._blocks([[0.3, 0.5]] * 25)
        b = self._blocks([[0.1, -0.1]] * 25)
        r1 = stats.block_bootstrap_diff_ci(a, b, n_boot=300, seed=7)
        r2 = stats.block_bootstrap_diff_ci(a, b, n_boot=300, seed=7)
        self.assertEqual(r1["ci_low"], r2["ci_low"])
        self.assertEqual(r1["ci_high"], r2["ci_high"])


class TestZeroCrossingVerdict(unittest.TestCase):
    def test_calls(self):
        self.assertEqual(stats.zero_crossing_verdict(0.1, 0.5), "edge")
        self.assertEqual(stats.zero_crossing_verdict(-0.5, -0.1), "adverse")
        self.assertEqual(stats.zero_crossing_verdict(-0.2, 0.3), "coin_flip")
        self.assertEqual(stats.zero_crossing_verdict(None, 0.3), "no_data")


class TestBenjaminiHochberg(unittest.TestCase):
    def test_all_reject_when_all_significant(self):
        self.assertEqual(stats.benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05], q=0.05),
                         [True, True, True, True, True])

    def test_partial_rejection_at_step_up(self):
        # sorted 0.01,0.20,0.90; m=3; thr 0.0167/0.033/0.05 → only 0.01 passes.
        self.assertEqual(stats.benjamini_hochberg([0.20, 0.90, 0.01], q=0.05),
                         [False, False, True])

    def test_none_pvalues_never_reject(self):
        self.assertEqual(stats.benjamini_hochberg([0.001, None], q=0.05), [True, False])

    def test_step_up_property_rescues_middle(self):
        # BH rejects up to the LARGEST passing rank, even if an earlier rank failed.
        # sorted 0.001,0.04,0.045; m=3; thr .0167/.033/.05 → rank3 0.045<=.05 → k=3, all reject.
        self.assertEqual(stats.benjamini_hochberg([0.045, 0.001, 0.04], q=0.05),
                         [True, True, True])


if __name__ == "__main__":
    unittest.main()


class TestSharpeMoments(unittest.TestCase):
    def test_known_series(self):
        # mean 0, sample std 1 exactly → sr 0; symmetric → skew 0.
        m = stats.sharpe_moments([-1.0, 0.0, 1.0])
        self.assertAlmostEqual(m["sr"], 0.0)
        self.assertAlmostEqual(m["skew"], 0.0)
        self.assertEqual(m["t"], 3)

    def test_normalish_kurtosis_near_three(self):
        import random as _r
        rng = _r.Random(11)
        m = stats.sharpe_moments([rng.gauss(0, 1) for _ in range(20000)])
        self.assertAlmostEqual(m["kurt"], 3.0, delta=0.15)
        self.assertAlmostEqual(m["skew"], 0.0, delta=0.1)

    def test_degenerate_none(self):
        self.assertIsNone(stats.sharpe_moments([1.0, 2.0])["sr"])       # too short
        self.assertIsNone(stats.sharpe_moments([2.0, 2.0, 2.0])["sr"])  # zero variance


class TestProbabilisticSharpe(unittest.TestCase):
    def test_hand_computed_normal_case(self):
        # sr=0.1, t=101, skew=0, kurt=3 → z = 0.1·√100/√1.005 = 0.99751 → Φ ≈ 0.84074
        psr = stats.probabilistic_sharpe(0.1, 101, 0.0, 3.0)
        self.assertAlmostEqual(psr, 0.84074, delta=1e-4)

    def test_positive_sr_beats_half_and_grows_with_t(self):
        p_small = stats.probabilistic_sharpe(0.1, 50, 0.0, 3.0)
        p_big = stats.probabilistic_sharpe(0.1, 500, 0.0, 3.0)
        self.assertGreater(p_small, 0.5)
        self.assertGreater(p_big, p_small)

    def test_fat_tails_shrink_confidence(self):
        thin = stats.probabilistic_sharpe(0.2, 200, 0.0, 3.0)
        fat = stats.probabilistic_sharpe(0.2, 200, -1.0, 10.0)  # neg skew + fat tails
        self.assertLess(fat, thin)

    def test_out_of_domain_none(self):
        self.assertIsNone(stats.probabilistic_sharpe(0.1, 1, 0.0, 3.0))       # t < 2
        self.assertIsNone(stats.probabilistic_sharpe(2.0, 100, 5.0, 3.0))     # denom ≤ 0


class TestExpectedMaxSharpe(unittest.TestCase):
    def test_no_selection_no_hurdle(self):
        self.assertEqual(stats.expected_max_sharpe(1, 1.0), 0.0)
        self.assertEqual(stats.expected_max_sharpe(10, 0.0), 0.0)

    def test_bracketed_by_the_two_quantiles(self):
        # Convex combination of Φ⁻¹(0.9)≈1.2816 and Φ⁻¹(1−1/(10e))≈1.79 at var=1.
        v = stats.expected_max_sharpe(10, 1.0)
        self.assertGreater(v, 1.28)
        self.assertLess(v, 1.79)

    def test_monotone_in_trials_and_scale(self):
        self.assertGreater(stats.expected_max_sharpe(100, 1.0),
                           stats.expected_max_sharpe(10, 1.0))
        self.assertAlmostEqual(stats.expected_max_sharpe(10, 4.0),
                               2 * stats.expected_max_sharpe(10, 1.0))


class TestDeflatedSharpe(unittest.TestCase):
    def _returns(self, mean, n=250, seed=7):
        import random as _r
        rng = _r.Random(seed)
        return [rng.gauss(mean, 1.0) for _ in range(n)]

    def test_single_trial_dsr_equals_psr(self):
        r = self._returns(0.1)
        out = stats.deflated_sharpe(r, n_trials=1, var_trials=0.0)
        self.assertEqual(out["sr_star"], 0.0)
        self.assertEqual(out["dsr"], out["psr"])

    def test_selection_always_costs(self):
        r = self._returns(0.1)
        out = stats.deflated_sharpe(r, n_trials=50, var_trials=0.01)
        self.assertLess(out["dsr"], out["psr"])

    def test_more_trials_more_deflation(self):
        r = self._returns(0.1)
        few = stats.deflated_sharpe(r, n_trials=5, var_trials=0.01)
        many = stats.deflated_sharpe(r, n_trials=500, var_trials=0.01)
        self.assertLess(many["dsr"], few["dsr"])

    def test_degenerate_series_none(self):
        out = stats.deflated_sharpe([1.0, 1.0, 1.0], n_trials=10, var_trials=0.01)
        self.assertIsNone(out["dsr"])


class TestPboCscv(unittest.TestCase):
    def _noise(self, n_trials, n_periods, seed):
        import random as _r
        rng = _r.Random(seed)
        return [[rng.gauss(0, 1) for _ in range(n_periods)] for _ in range(n_trials)]

    def test_pure_noise_pbo_near_half(self):
        m = self._noise(20, 400, seed=3)
        out = stats.pbo_cscv(m, n_blocks=8)
        self.assertEqual(out["n_combos"], 70)
        self.assertGreater(out["pbo"], 0.2)
        self.assertLess(out["pbo"], 0.8)

    def test_planted_edge_keeps_winning(self):
        import random as _r
        rng = _r.Random(5)
        m = self._noise(20, 400, seed=6)
        m[0] = [rng.gauss(0.4, 1) for _ in range(400)]  # one real edge
        out = stats.pbo_cscv(m, n_blocks=8, focus=0)
        self.assertLessEqual(out["pbo"], 0.1)
        self.assertGreaterEqual(out["focus"]["picked_frac"], 0.9)
        self.assertLessEqual(out["prob_oos_loss"], 0.1)

    def test_flat_trial_never_selected(self):
        m = self._noise(5, 200, seed=9)
        m[2] = [0.0] * 200                      # flat → -inf Sharpe, must not win
        out = stats.pbo_cscv(m, n_blocks=4, focus=2)
        self.assertIsNotNone(out["pbo"])
        self.assertEqual(out["focus"]["picked_frac"], 0.0)

    def test_explicit_block_ids_match_default(self):
        m = self._noise(6, 120, seed=4)
        base, rem = divmod(120, 6)
        ids = []
        for b in range(6):
            ids.extend([b] * (base + (1 if b < rem else 0)))
        d = stats.pbo_cscv(m, n_blocks=6)
        e = stats.pbo_cscv(m, n_blocks=6, block_ids=ids)
        self.assertEqual(d["pbo"], e["pbo"])
        self.assertEqual(d["median_logit"], e["median_logit"])

    def test_degenerate_inputs_none(self):
        self.assertIsNone(stats.pbo_cscv([[1.0] * 50], n_blocks=4)["pbo"])          # 1 trial
        m = self._noise(4, 100, seed=1)
        self.assertIsNone(stats.pbo_cscv(m, n_blocks=5)["pbo"])                     # odd blocks
        self.assertIsNone(stats.pbo_cscv(m, n_blocks=4, block_ids=[0] * 99)["pbo"])  # bad ids
