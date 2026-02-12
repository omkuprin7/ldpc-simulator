"""Structured result data model for LDPC simulations with JSON/CSV export."""

import json
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


@dataclass
class BlockResult:
    """Per-block result from a single encode/decode cycle."""
    block_num: int
    snr_db: float
    decode_success: bool
    error_bits: int
    normalized_llr: float
    convergence_iteration: int  # iteration when syndrome=0, or -1 if failed


@dataclass
class SNRPointResult:
    """Aggregated results for a single SNR point."""
    snr_db: float
    ber: float
    fer: float
    avg_normalized_llr: float
    total_blocks: int
    successful_blocks: int
    failed_blocks: int
    avg_convergence_iterations: float
    # Parameters used for this SNR point (tracks adaptive changes)
    matrix_path: str = ""
    modulation: int = 1
    max_iterations: int = 5
    interleaver: str = "none"
    encoding_method: str = "standard"


@dataclass
class SimulationConfig:
    """Captures all parameters of a simulation run."""
    matrix_path: str
    n: int
    m: int
    k: int
    rate: float
    blocks: int
    max_iterations: int
    encoding_method: str
    interleaver_type: str
    decoder_type: str
    channel_mode: int
    modulation: int
    speed: float
    snr_range: Tuple[float, float, float]  # (start, end, step)
    threads: int
    timestamp: str
    interference_snr: float = 0.0
    p: float = 0.1


@dataclass
class SimulationResult:
    """Complete simulation result container."""
    config: SimulationConfig
    snr_points: List[SNRPointResult]
    wall_clock_seconds: float
    adaptation_log: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        # Convert tuple to list for JSON
        d['config']['snr_range'] = list(d['config']['snr_range'])
        return d

    def to_json(self, filepath: str) -> None:
        """Export results to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def to_csv(self, filepath: str) -> None:
        """Export per-SNR results to a CSV file."""
        if not self.snr_points:
            return
        fieldnames = [
            'snr_db', 'ber', 'fer', 'avg_normalized_llr',
            'total_blocks', 'successful_blocks', 'failed_blocks',
            'avg_convergence_iterations',
            'matrix_path', 'modulation', 'max_iterations',
            'interleaver', 'encoding_method'
        ]
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sp in self.snr_points:
                row = {k: getattr(sp, k) for k in fieldnames}
                writer.writerow(row)

    @classmethod
    def from_json(cls, filepath: str) -> 'SimulationResult':
        """Load results from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)

        config_d = d['config']
        config_d['snr_range'] = tuple(config_d['snr_range'])
        config = SimulationConfig(**config_d)

        snr_points = [SNRPointResult(**sp) for sp in d['snr_points']]

        return cls(
            config=config,
            snr_points=snr_points,
            wall_clock_seconds=d['wall_clock_seconds'],
            adaptation_log=d.get('adaptation_log', [])
        )
