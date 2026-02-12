"""Shared fixtures for LDPC simulation tests."""

import os
import sys
import pytest

# Ensure python_ldpc_app is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def bch_matrix_path():
    """Path to the small BCH(7,4) test matrix."""
    return os.path.join(
        os.path.dirname(__file__), '..', '..',
        'Channel_Codes_Database', 'BCH_7_4_1_strip.alist.txt'
    )


@pytest.fixture
def matrix_catalog_path():
    """Path to the Channel_Codes_Database directory."""
    return os.path.join(
        os.path.dirname(__file__), '..', '..', 'Channel_Codes_Database'
    )


@pytest.fixture
def sample_simulation_result():
    """A synthetic SimulationResult for testing visualization and export."""
    from results import SimulationResult, SimulationConfig, SNRPointResult

    config = SimulationConfig(
        matrix_path='/test/BCH_7_4_1_strip.alist.txt',
        n=7, m=3, k=4, rate=4/7,
        blocks=10, max_iterations=5,
        encoding_method='standard',
        interleaver_type='none',
        decoder_type='sumproduct',
        channel_mode=1, modulation=1, speed=1.0,
        snr_range=(0.0, 2.0, 1.0),
        threads=1,
        timestamp='2026-01-01T00:00:00',
    )

    snr_points = [
        SNRPointResult(
            snr_db=0.0, ber=0.15, fer=0.8,
            avg_normalized_llr=0.25,
            total_blocks=10, successful_blocks=2, failed_blocks=8,
            avg_convergence_iterations=4.5,
        ),
        SNRPointResult(
            snr_db=1.0, ber=0.05, fer=0.4,
            avg_normalized_llr=0.12,
            total_blocks=10, successful_blocks=6, failed_blocks=4,
            avg_convergence_iterations=3.2,
        ),
        SNRPointResult(
            snr_db=2.0, ber=0.001, fer=0.1,
            avg_normalized_llr=0.03,
            total_blocks=10, successful_blocks=9, failed_blocks=1,
            avg_convergence_iterations=1.8,
        ),
    ]

    return SimulationResult(
        config=config,
        snr_points=snr_points,
        wall_clock_seconds=1.5,
    )
