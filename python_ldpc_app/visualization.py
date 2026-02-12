"""Visualization module for LDPC simulation results using matplotlib."""

import os

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend by default
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from results import SimulationResult


class SimulationPlotter:
    """Generates standard LDPC performance plots from SimulationResult data."""

    def __init__(self, result: SimulationResult):
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )
        self.result = result

    def _get_snr_values(self):
        return [sp.snr_db for sp in self.result.snr_points]

    def plot_ber_vs_snr(self, ax=None, save_path=None, label=None):
        """BER vs SNR waterfall curve (semilogy)."""
        snrs = self._get_snr_values()
        bers = [sp.ber for sp in self.result.snr_points]

        # Filter zero BER values (can't plot on log scale)
        plot_snrs = [s for s, b in zip(snrs, bers) if b > 0]
        plot_bers = [b for b in bers if b > 0]

        if not plot_snrs:
            return ax

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 6))

        lbl = label or f"Rate={self.result.config.rate:.3f}"
        ax.semilogy(plot_snrs, plot_bers, 'o-', label=lbl, markersize=5)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')
        ax.set_title('BER vs SNR')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()

        if save_path and own_fig:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return ax

    def plot_fer_vs_snr(self, ax=None, save_path=None, label=None):
        """FER vs SNR waterfall curve (semilogy)."""
        snrs = self._get_snr_values()
        fers = [sp.fer for sp in self.result.snr_points]

        plot_snrs = [s for s, f in zip(snrs, fers) if f > 0]
        plot_fers = [f for f in fers if f > 0]

        if not plot_snrs:
            return ax

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 6))

        lbl = label or f"Rate={self.result.config.rate:.3f}"
        ax.semilogy(plot_snrs, plot_fers, 's-', label=lbl, markersize=5)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('FER')
        ax.set_title('FER vs SNR')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()

        if save_path and own_fig:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return ax

    def plot_llr_vs_snr(self, ax=None, save_path=None, label=None):
        """Normalized LLR vs SNR curve."""
        snrs = self._get_snr_values()
        llrs = [sp.avg_normalized_llr for sp in self.result.snr_points]

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 6))

        lbl = label or f"Rate={self.result.config.rate:.3f}"
        ax.plot(snrs, llrs, 'd-', label=lbl, markersize=5)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Normalized LLR')
        ax.set_title('Normalized LLR vs SNR')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path and own_fig:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return ax

    def plot_convergence_vs_snr(self, ax=None, save_path=None, label=None):
        """Average decoder convergence iterations vs SNR."""
        snrs = self._get_snr_values()
        iters = [sp.avg_convergence_iterations for sp in self.result.snr_points]

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 6))

        lbl = label or f"Rate={self.result.config.rate:.3f}"
        ax.plot(snrs, iters, '^-', label=lbl, markersize=5)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Avg iterations to convergence')
        ax.set_title('Decoder Convergence vs SNR')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path and own_fig:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return ax

    def plot_combined_dashboard(self, save_dir=None):
        """2x2 subplot grid: BER, FER, LLR, convergence."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'LDPC Simulation: {os.path.basename(self.result.config.matrix_path)} '
            f'(n={self.result.config.n}, k={self.result.config.k}, rate={self.result.config.rate:.3f})',
            fontsize=13
        )

        self.plot_ber_vs_snr(ax=axes[0, 0])
        self.plot_fer_vs_snr(ax=axes[0, 1])
        self.plot_llr_vs_snr(ax=axes[1, 0])
        self.plot_convergence_vs_snr(ax=axes[1, 1])

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, 'dashboard.png'),
                dpi=150, bbox_inches='tight'
            )

        return fig

    def plot_adaptation_history(self, save_dir=None):
        """Plot parameter changes over SNR for adaptive simulations."""
        if not self.result.adaptation_log:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Adaptive Parameter History', fontsize=13)

        snrs = [entry['snr_db'] for entry in self.result.adaptation_log]
        rates = [entry.get('rate', 0) for entry in self.result.adaptation_log]
        iters = [entry.get('max_iterations', 0) for entry in self.result.adaptation_log]

        axes[0].plot(snrs, rates, 'o-', color='tab:blue', markersize=6)
        axes[0].set_xlabel('SNR (dB)')
        axes[0].set_ylabel('Code Rate')
        axes[0].set_title('Code Rate vs SNR')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(snrs, iters, 's-', color='tab:orange', markersize=6)
        axes[1].set_xlabel('SNR (dB)')
        axes[1].set_ylabel('Max Iterations')
        axes[1].set_title('Max Decoder Iterations vs SNR')
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                os.path.join(save_dir, 'adaptation_history.png'),
                dpi=150, bbox_inches='tight'
            )

        return fig

    @staticmethod
    def plot_comparison(results, metric='ber', save_path=None):
        """Overlay multiple SimulationResult objects on the same plot.

        Args:
            results: list of SimulationResult
            metric: 'ber', 'fer', 'llr', or 'convergence'
            save_path: optional path to save the figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization.")

        fig, ax = plt.subplots(figsize=(10, 7))

        for r in results:
            plotter = SimulationPlotter(r)
            label = f"{os.path.basename(r.config.matrix_path)} (rate={r.config.rate:.3f})"
            if metric == 'ber':
                plotter.plot_ber_vs_snr(ax=ax, label=label)
            elif metric == 'fer':
                plotter.plot_fer_vs_snr(ax=ax, label=label)
            elif metric == 'llr':
                plotter.plot_llr_vs_snr(ax=ax, label=label)
            elif metric == 'convergence':
                plotter.plot_convergence_vs_snr(ax=ax, label=label)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
