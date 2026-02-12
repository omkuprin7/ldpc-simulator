# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python console application for LDPC (Low-Density Parity-Check) code simulation. Provides encoding, channel simulation (AWGN with optional interference), and Sum-Product Algorithm (SPA) decoding.

## Running the Application

```bash
cd python_ldpc_app
pip install -r requirements.txt

# Basic run
python main.py --matrix ../Channel_Codes_Database/BCH_7_4_1_strip.alist.txt

# Full simulation with SNR sweep and multiprocessing
python main.py --matrix ../Channel_Codes_Database/BCH_7_4_1_strip.alist.txt \
    --initial-snr 0.0 --end-snr 5.0 --step-snr 0.5 \
    --blocks 1000 --threads 4 --ber --fer
```

## Key CLI Arguments

- `--matrix, -m`: Path to ALIST format parity check matrix (required)
- `--blocks, -b`: Number of codewords to simulate (default: 100)
- `--iterations, -i`: Max decoder iterations (default: 5)
- `--threads, -t`: Number of processes for parallel processing (default: 1)
- `--mode`: Channel mode - 1=AWGN, 2=AWGN+partial-band interference, 3=AWGN+jamming
- `--encoding-method`: `standard` or `richardson-urbanke`
- `--interleaver, -il`: none, regular, random, srandom
- `--ber`, `--fer`, `--normalized-llr`: Enable specific metrics

## Architecture

**Processing Pipeline:** Data → Encoding → Interleaving → Channel (noise) → Deinterleaving → SPA Decoding

**Core Modules (in `python_ldpc_app/`):**
- `main.py` - Entry point, CLI parsing, multiprocess orchestration
- `encoder_decoder_data.py` - Matrix loading, generator matrix computation via Gaussian elimination
- `spa_decoder.py` - Sum-Product Algorithm with sparse matrix operations
- `channel.py` - AWGN and interference channel models
- `data_buffer.py` - Data container through encode/decode stages
- `matrix_sparse.py` - scipy.sparse wrapper for efficient LDPC operations
- `interleavers.py` - Regular, random, and S-random interleaver implementations
- `utils.py` - ALIST file format parser

**Multiprocessing:** Uses `ProcessPoolExecutor` (not threads) to bypass Python's GIL. Each process gets independent decoder/channel instances with shared read-only matrix data.

## Matrix Format

Matrices use ALIST format (standard LDPC format):
- Line 1: `N M` (columns, rows)
- Line 2: Max column/row weights
- Lines 3-4: Individual column/row weights
- Remaining lines: Non-zero positions per column, then per row
