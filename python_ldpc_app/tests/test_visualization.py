"""Tests for visualization.py: plot generation without errors."""

import os
import tempfile
import pytest

# Force non-interactive backend before importing visualization
import matplotlib
matplotlib.use('Agg')

from visualization import SimulationPlotter


class TestSimulationPlotter:
    def test_create_plotter(self, sample_simulation_result):
        plotter = SimulationPlotter(sample_simulation_result)
        assert plotter.result is sample_simulation_result

    def test_plot_ber_vs_snr(self, sample_simulation_result):
        plotter = SimulationPlotter(sample_simulation_result)
        ax = plotter.plot_ber_vs_snr()
        assert ax is not None

    def test_plot_fer_vs_snr(self, sample_simulation_result):
        plotter = SimulationPlotter(sample_simulation_result)
        ax = plotter.plot_fer_vs_snr()
        assert ax is not None

    def test_plot_llr_vs_snr(self, sample_simulation_result):
        plotter = SimulationPlotter(sample_simulation_result)
        ax = plotter.plot_llr_vs_snr()
        assert ax is not None

    def test_plot_convergence_vs_snr(self, sample_simulation_result):
        plotter = SimulationPlotter(sample_simulation_result)
        ax = plotter.plot_convergence_vs_snr()
        assert ax is not None

    def test_plot_combined_dashboard(self, sample_simulation_result):
        plotter = SimulationPlotter(sample_simulation_result)
        fig = plotter.plot_combined_dashboard()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_dashboard_save(self, sample_simulation_result):
        plotter = SimulationPlotter(sample_simulation_result)
        with tempfile.TemporaryDirectory() as tmpdir:
            fig = plotter.plot_combined_dashboard(save_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, 'dashboard.png'))
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_plot_ber_save(self, sample_simulation_result):
        plotter = SimulationPlotter(sample_simulation_result)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'ber.png')
            plotter.plot_ber_vs_snr(save_path=path)
            assert os.path.exists(path)

    def test_plot_comparison(self, sample_simulation_result):
        fig = SimulationPlotter.plot_comparison(
            [sample_simulation_result, sample_simulation_result],
            metric='ber'
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_adaptation_history_empty(self, sample_simulation_result):
        """No adaptation log -> returns None."""
        plotter = SimulationPlotter(sample_simulation_result)
        fig = plotter.plot_adaptation_history()
        assert fig is None

    def test_plot_adaptation_history_with_data(self, sample_simulation_result):
        sample_simulation_result.adaptation_log = [
            {'snr_db': 0.0, 'rate': 0.5, 'max_iterations': 5},
            {'snr_db': 1.0, 'rate': 0.66, 'max_iterations': 10},
            {'snr_db': 2.0, 'rate': 0.75, 'max_iterations': 10},
        ]
        plotter = SimulationPlotter(sample_simulation_result)
        fig = plotter.plot_adaptation_history()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
