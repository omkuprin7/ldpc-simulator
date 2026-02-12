"""Catalog of available LDPC matrices with metadata for adaptive parameter selection."""

import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MatrixInfo:
    """Metadata about an ALIST matrix file."""
    path: str
    name: str
    n: int
    k: int
    m: int
    rate: float
    family: str  # 'wimax', 'ccsds', 'bch', 'wifi', 'wran', 'wigig', 'custom'


class MatrixCatalog:
    """Registry of available LDPC matrices, indexed by properties."""

    def __init__(self, base_dir: str):
        self.matrices: List[MatrixInfo] = []
        self._scan_directory(base_dir)
        # Sort by rate for easy navigation
        self.matrices.sort(key=lambda m: (m.family, m.rate, m.n))

    def _scan_directory(self, base_dir: str) -> None:
        """Recursively scan for ALIST files and parse metadata from filenames."""
        for root, _dirs, files in os.walk(base_dir):
            for fname in files:
                if not fname.endswith('.alist.txt') and not fname.endswith('.txt'):
                    continue
                filepath = os.path.join(root, fname)
                info = self._parse_filename(filepath, fname)
                if info:
                    self.matrices.append(info)

    def _parse_filename(self, filepath: str, fname: str) -> Optional[MatrixInfo]:
        """Try to extract matrix parameters from the filename.
        Falls back to reading the ALIST header if filename parsing fails."""

        # Pattern: wimax_576_0.5.alist.txt
        m = re.match(r'wimax_(\d+)_([\d.]+[A-B]?)\.alist\.txt', fname)
        if m:
            n = int(m.group(1))
            rate_str = m.group(2)
            # Handle rate strings like '0.66B', '0.75A'
            rate = float(re.sub(r'[A-Za-z]', '', rate_str))
            k = int(round(n * rate))
            return MatrixInfo(
                path=filepath, name=fname, n=n, k=k, m=n - k,
                rate=rate, family='wimax'
            )

        # Pattern: CCSDS_ldpc_n128_k64.alist.txt
        m = re.match(r'CCSDS_ldpc_n(\d+)_k(\d+)\.alist\.txt', fname)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            return MatrixInfo(
                path=filepath, name=fname, n=n, k=k, m=n - k,
                rate=k / n if n > 0 else 0, family='ccsds'
            )

        # Pattern: wifi_648_r083.alist.txt
        m = re.match(r'wifi_(\d+)_r(\d+)\.alist\.txt', fname)
        if m:
            n = int(m.group(1))
            rate = int(m.group(2)) / 100.0
            k = int(round(n * rate))
            return MatrixInfo(
                path=filepath, name=fname, n=n, k=k, m=n - k,
                rate=rate, family='wifi'
            )

        # Pattern: wigig_R05_N672_K336.alist.txt
        m = re.match(r'wigig_R(\d+)_N(\d+)_K(\d+)\.alist\.txt', fname)
        if m:
            rate = int(m.group(1)) / 100.0
            n, k = int(m.group(2)), int(m.group(3))
            return MatrixInfo(
                path=filepath, name=fname, n=n, k=k, m=n - k,
                rate=k / n if n > 0 else rate, family='wigig'
            )

        # Pattern: WRAN_N384_K192_P16_R05.txt
        m = re.match(r'WRAN_N(\d+)_K(\d+)_P\d+_R(\d+)\.txt', fname)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            return MatrixInfo(
                path=filepath, name=fname, n=n, k=k, m=n - k,
                rate=k / n if n > 0 else 0, family='wran'
            )

        # Pattern: BCH_7_4_1_strip.alist.txt
        m = re.match(r'BCH_(\d+)_(\d+)_\d+', fname)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            return MatrixInfo(
                path=filepath, name=fname, n=n, k=k, m=n - k,
                rate=k / n if n > 0 else 0, family='bch'
            )

        # Pattern: Tanner_155_64.alist.txt
        m = re.match(r'Tanner_(\d+)_(\d+)\.alist\.txt', fname)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            return MatrixInfo(
                path=filepath, name=fname, n=n, k=k, m=n - k,
                rate=k / n if n > 0 else 0, family='custom'
            )

        # Pattern: ieee_802_11ad_... or LDPC_N336_K196_...
        m = re.match(r'LDPC_N(\d+)_K(\d+)', fname)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            return MatrixInfo(
                path=filepath, name=fname, n=n, k=k, m=n - k,
                rate=k / n if n > 0 else 0, family='custom'
            )

        # Fallback: try reading the ALIST header
        return self._parse_alist_header(filepath, fname)

    def _parse_alist_header(self, filepath: str, fname: str) -> Optional[MatrixInfo]:
        """Read the first line of an ALIST file to get N and M."""
        try:
            with open(filepath, 'r') as f:
                line = f.readline().strip()
            parts = line.split()
            if len(parts) >= 2:
                n, m_val = int(parts[0]), int(parts[1])
                k = n - m_val
                return MatrixInfo(
                    path=filepath, name=fname, n=n, k=k, m=m_val,
                    rate=k / n if n > 0 else 0, family='unknown'
                )
        except (ValueError, IOError):
            pass
        return None

    def get_by_rate_range(self, min_rate: float, max_rate: float) -> List[MatrixInfo]:
        """Return matrices within the given rate range."""
        return [m for m in self.matrices if min_rate <= m.rate <= max_rate]

    def get_by_family(self, family: str) -> List[MatrixInfo]:
        """Return matrices of a specific family."""
        return [m for m in self.matrices if m.family == family]

    def get_nearest_rate(self, target_rate: float, family: str = None,
                         block_size: int = None) -> Optional[MatrixInfo]:
        """Find the matrix closest to target_rate, optionally filtered by family/size."""
        candidates = self.matrices
        if family:
            candidates = [m for m in candidates if m.family == family]
        if block_size:
            candidates = [m for m in candidates if m.n == block_size]
        if not candidates:
            return None
        return min(candidates, key=lambda m: abs(m.rate - target_rate))

    def get_lower_rate(self, current: MatrixInfo) -> Optional[MatrixInfo]:
        """Find the next lower-rate matrix in the same family and block size."""
        candidates = [
            m for m in self.matrices
            if m.family == current.family and m.n == current.n and m.rate < current.rate
        ]
        if not candidates:
            # Try same family, any block size
            candidates = [
                m for m in self.matrices
                if m.family == current.family and m.rate < current.rate
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.rate)  # closest lower rate

    def get_higher_rate(self, current: MatrixInfo) -> Optional[MatrixInfo]:
        """Find the next higher-rate matrix in the same family and block size."""
        candidates = [
            m for m in self.matrices
            if m.family == current.family and m.n == current.n and m.rate > current.rate
        ]
        if not candidates:
            candidates = [
                m for m in self.matrices
                if m.family == current.family and m.rate > current.rate
            ]
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.rate)  # closest higher rate

    def __len__(self):
        return len(self.matrices)

    def __repr__(self):
        families = {}
        for m in self.matrices:
            families[m.family] = families.get(m.family, 0) + 1
        parts = [f"{f}={c}" for f, c in sorted(families.items())]
        return f"MatrixCatalog({len(self.matrices)} matrices: {', '.join(parts)})"
