#!/usr/bin/env python3
"""Standalone script for plotting previously saved LDPC simulation results.

Usage:
    python plot_results.py result1.json result2.json --metric ber --output comparison.png
    python plot_results.py result.json --dashboard --output-dir ./plots
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results import SimulationResult


def main():
    parser = argparse.ArgumentParser(
        description='Plot LDPC simulation results from JSON files'
    )
    parser.add_argument('files', nargs='+', help='JSON result files to plot')
    parser.add_argument('--metric', type=str, choices=['ber', 'fer', 'llr', 'convergence'],
                        default='ber', help='Metric to plot for comparison (default: ber)')
    parser.add_argument('--dashboard', action='store_true',
                        help='Show combined dashboard for each result file')
    parser.add_argument('--output', type=str, default=None,
                        help='Save comparison plot to file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Save dashboard plots to directory')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not show interactive plots')

    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    from visualization import SimulationPlotter

    results = []
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Warning: file not found: {filepath}")
            continue
        results.append(SimulationResult.from_json(filepath))

    if not results:
        print("Error: no valid result files loaded")
        sys.exit(1)

    if args.dashboard:
        for i, r in enumerate(results):
            plotter = SimulationPlotter(r)
            plotter.plot_combined_dashboard(save_dir=args.output_dir)
            if r.adaptation_log:
                plotter.plot_adaptation_history(save_dir=args.output_dir)

    if len(results) >= 1:
        SimulationPlotter.plot_comparison(
            results, metric=args.metric, save_path=args.output
        )

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
