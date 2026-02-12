"""Tests for results.py: serialization, JSON/CSV round-trip."""

import json
import os
import tempfile
import pytest

from results import SimulationResult, SimulationConfig, SNRPointResult, BlockResult


class TestBlockResult:
    def test_creation(self):
        br = BlockResult(
            block_num=0, snr_db=1.0, decode_success=True,
            error_bits=0, normalized_llr=0.1, convergence_iteration=3
        )
        assert br.decode_success is True
        assert br.convergence_iteration == 3

    def test_failed_block(self):
        br = BlockResult(
            block_num=1, snr_db=0.0, decode_success=False,
            error_bits=2, normalized_llr=0.5, convergence_iteration=-1
        )
        assert br.decode_success is False
        assert br.convergence_iteration == -1


class TestSNRPointResult:
    def test_defaults(self):
        sp = SNRPointResult(
            snr_db=1.0, ber=0.05, fer=0.4,
            avg_normalized_llr=0.1,
            total_blocks=100, successful_blocks=60, failed_blocks=40,
            avg_convergence_iterations=3.0,
        )
        assert sp.matrix_path == ""
        assert sp.modulation == 1
        assert sp.max_iterations == 5

    def test_custom_params(self):
        sp = SNRPointResult(
            snr_db=2.0, ber=0.01, fer=0.2,
            avg_normalized_llr=0.05,
            total_blocks=50, successful_blocks=40, failed_blocks=10,
            avg_convergence_iterations=2.5,
            matrix_path='/test.alist',
            modulation=2,
            max_iterations=10,
            interleaver='random',
            encoding_method='richardson-urbanke',
        )
        assert sp.modulation == 2
        assert sp.interleaver == 'random'


class TestSimulationResult:
    def test_to_dict(self, sample_simulation_result):
        d = sample_simulation_result.to_dict()
        assert isinstance(d, dict)
        assert 'config' in d
        assert 'snr_points' in d
        assert len(d['snr_points']) == 3
        # snr_range should be a list (not tuple) in dict form
        assert isinstance(d['config']['snr_range'], list)

    def test_json_roundtrip(self, sample_simulation_result):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name

        try:
            sample_simulation_result.to_json(path)
            loaded = SimulationResult.from_json(path)

            assert loaded.config.n == sample_simulation_result.config.n
            assert loaded.config.k == sample_simulation_result.config.k
            assert loaded.config.rate == pytest.approx(sample_simulation_result.config.rate)
            assert len(loaded.snr_points) == len(sample_simulation_result.snr_points)

            for orig, load in zip(sample_simulation_result.snr_points, loaded.snr_points):
                assert load.snr_db == pytest.approx(orig.snr_db)
                assert load.ber == pytest.approx(orig.ber)
                assert load.fer == pytest.approx(orig.fer)

            assert loaded.wall_clock_seconds == pytest.approx(
                sample_simulation_result.wall_clock_seconds
            )
        finally:
            os.unlink(path)

    def test_csv_export(self, sample_simulation_result):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            path = f.name

        try:
            sample_simulation_result.to_csv(path)
            with open(path, 'r') as f:
                lines = f.readlines()
            # Header + 3 data rows
            assert len(lines) == 4
            assert 'snr_db' in lines[0]
            assert 'ber' in lines[0]
        finally:
            os.unlink(path)

    def test_empty_result(self):
        config = SimulationConfig(
            matrix_path='/test.alist',
            n=7, m=3, k=4, rate=4/7,
            blocks=0, max_iterations=5,
            encoding_method='standard',
            interleaver_type='none',
            decoder_type='sumproduct',
            channel_mode=1, modulation=1, speed=1.0,
            snr_range=(0.0, 0.0, 1.0),
            threads=1,
            timestamp='2026-01-01T00:00:00',
        )
        result = SimulationResult(config=config, snr_points=[], wall_clock_seconds=0.0)
        d = result.to_dict()
        assert len(d['snr_points']) == 0
