"""Tests for adaptive.py: strategy evaluation with mock data."""

import pytest

from adaptive import ThresholdStrategy, AdaptiveState, AdaptiveAction
from results import SNRPointResult


def _make_state(**overrides):
    defaults = dict(
        current_matrix_path='/test/wimax_576_0.5.alist.txt',
        current_rate=0.5,
        current_modulation=1,
        current_max_iterations=5,
        current_interleaver='none',
        current_encoding_method='standard',
    )
    defaults.update(overrides)
    return AdaptiveState(**defaults)


def _make_snr_result(**overrides):
    defaults = dict(
        snr_db=1.0, ber=0.05, fer=0.4,
        avg_normalized_llr=0.1,
        total_blocks=100, successful_blocks=60, failed_blocks=40,
        avg_convergence_iterations=3.0,
    )
    defaults.update(overrides)
    return SNRPointResult(**defaults)


class TestThresholdStrategy:
    def test_high_ber_triggers_lower_rate(self):
        strategy = ThresholdStrategy(high_ber_threshold=1e-2)
        state = _make_state()
        result = _make_snr_result(ber=0.05)  # > 1e-2
        action = strategy.evaluate(state, result)
        assert action is not None
        assert action.new_matrix_path == "__LOWER_RATE__"
        assert "switching to lower rate" in action.reason

    def test_low_ber_triggers_higher_rate(self):
        strategy = ThresholdStrategy(low_ber_threshold=1e-5)
        state = _make_state()
        result = _make_snr_result(ber=1e-6)
        action = strategy.evaluate(state, result)
        assert action is not None
        assert action.new_matrix_path == "__HIGHER_RATE__"
        assert "switching to higher rate" in action.reason

    def test_zero_ber_no_higher_rate(self):
        """Zero BER should not trigger higher rate (can't confirm it's genuinely zero)."""
        strategy = ThresholdStrategy(low_ber_threshold=1e-5)
        state = _make_state()
        result = _make_snr_result(ber=0.0)
        action = strategy.evaluate(state, result)
        # Zero BER doesn't trigger higher rate (ber > 0 check in strategy)
        assert action is None or action.new_matrix_path != "__HIGHER_RATE__"

    def test_normal_ber_no_action(self):
        strategy = ThresholdStrategy(high_ber_threshold=1e-2, low_ber_threshold=1e-5)
        state = _make_state()
        result = _make_snr_result(ber=1e-3)  # between thresholds
        action = strategy.evaluate(state, result)
        assert action is None

    def test_slow_convergence_increases_iterations(self):
        strategy = ThresholdStrategy(convergence_ratio=0.8)
        state = _make_state(current_max_iterations=5)
        # avg_convergence = 4.5 > 0.8 * 5 = 4.0
        result = _make_snr_result(ber=1e-3, avg_convergence_iterations=4.5)
        action = strategy.evaluate(state, result)
        assert action is not None
        assert action.new_max_iterations == 10  # doubled from 5

    def test_high_fer_enables_interleaver(self):
        strategy = ThresholdStrategy(fer_threshold=0.5)
        state = _make_state(current_interleaver='none')
        result = _make_snr_result(ber=1e-3, fer=0.6)
        action = strategy.evaluate(state, result)
        assert action is not None
        assert action.new_interleaver == 'random'

    def test_fer_with_existing_interleaver_no_change(self):
        strategy = ThresholdStrategy(fer_threshold=0.5)
        state = _make_state(current_interleaver='random')
        result = _make_snr_result(ber=1e-3, fer=0.6)
        action = strategy.evaluate(state, result)
        # Should not suggest interleaver change since already using one
        if action:
            assert action.new_interleaver is None

    def test_strategy_name(self):
        strategy = ThresholdStrategy()
        assert strategy.get_name() == "threshold"
