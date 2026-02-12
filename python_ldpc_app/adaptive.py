"""Adaptive parameter selection for LDPC simulations.

Implements a feedback loop that adjusts transmission parameters between SNR points
based on observed performance metrics (BER, FER, convergence speed).
"""

import os
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed

from encoder_decoder_data import EncoderDecoderData
from spa_decoder import SPA_Decoder
from channel import Channel
from settings import Settings
from enums import InterleaverType, LDPCDecoderType, Result, EncodingMethod
from results import SimulationResult, SimulationConfig, SNRPointResult
from matrix_catalog import MatrixCatalog, MatrixInfo


@dataclass
class AdaptiveState:
    """Current state of the adaptive controller."""
    current_matrix_path: str
    current_rate: float
    current_modulation: int
    current_max_iterations: int
    current_interleaver: str
    current_encoding_method: str
    history: List[dict] = field(default_factory=list)


@dataclass
class AdaptiveAction:
    """Describes a parameter change decided by a strategy."""
    new_matrix_path: Optional[str] = None
    new_modulation: Optional[int] = None
    new_max_iterations: Optional[int] = None
    new_interleaver: Optional[str] = None
    reason: str = ""


class AdaptiveStrategy(ABC):
    """Base class for adaptive parameter selection strategies."""

    @abstractmethod
    def evaluate(self, state: AdaptiveState,
                 last_snr_result: SNRPointResult) -> Optional[AdaptiveAction]:
        """Evaluate performance and decide on parameter changes.
        Returns None if no change needed, or an AdaptiveAction."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...


class ThresholdStrategy(AdaptiveStrategy):
    """Threshold-based adaptation between SNR points.

    Rules:
    - BER > high_ber_threshold  -> switch to lower rate code (more protection)
    - BER < low_ber_threshold   -> switch to higher rate code (more throughput)
    - avg_convergence_iters > 0.8 * max_iterations -> increase max_iterations
    - FER > fer_threshold -> try different interleaver
    """

    def __init__(self, high_ber_threshold=1e-2, low_ber_threshold=1e-5,
                 fer_threshold=0.5, convergence_ratio=0.8):
        self.high_ber_threshold = high_ber_threshold
        self.low_ber_threshold = low_ber_threshold
        self.fer_threshold = fer_threshold
        self.convergence_ratio = convergence_ratio

    def get_name(self) -> str:
        return "threshold"

    def evaluate(self, state: AdaptiveState,
                 last_snr_result: SNRPointResult) -> Optional[AdaptiveAction]:
        action = AdaptiveAction()
        reasons = []

        # Rule 1: BER too high -> need more protection (lower rate)
        if last_snr_result.ber > self.high_ber_threshold:
            action.new_matrix_path = "__LOWER_RATE__"  # Sentinel, resolved by controller
            reasons.append(
                f"BER={last_snr_result.ber:.2e} > {self.high_ber_threshold:.2e}, switching to lower rate"
            )

        # Rule 2: BER very low -> can increase throughput (higher rate)
        elif last_snr_result.ber < self.low_ber_threshold and last_snr_result.ber > 0:
            action.new_matrix_path = "__HIGHER_RATE__"
            reasons.append(
                f"BER={last_snr_result.ber:.2e} < {self.low_ber_threshold:.2e}, switching to higher rate"
            )

        # Rule 3: Decoder converging too slowly -> increase iterations
        if (last_snr_result.avg_convergence_iterations >
                self.convergence_ratio * state.current_max_iterations):
            new_iters = min(state.current_max_iterations * 2, 100)
            if new_iters > state.current_max_iterations:
                action.new_max_iterations = new_iters
                reasons.append(
                    f"avg_conv={last_snr_result.avg_convergence_iterations:.1f} near max={state.current_max_iterations}, "
                    f"increasing to {new_iters}"
                )

        # Rule 4: High FER with random interleaver might help
        if (last_snr_result.fer > self.fer_threshold and
                state.current_interleaver == 'none'):
            action.new_interleaver = 'random'
            reasons.append(
                f"FER={last_snr_result.fer:.3f} > {self.fer_threshold}, enabling random interleaver"
            )

        if not reasons:
            return None

        action.reason = "; ".join(reasons)
        return action


class AdaptiveController:
    """Orchestrates adaptive simulation across SNR points."""

    def __init__(self, strategy: AdaptiveStrategy, catalog: MatrixCatalog):
        self.strategy = strategy
        self.catalog = catalog
        self._encoder_cache = {}  # path -> EncoderDecoderData

    def _get_encoder_decoder_data(self, matrix_path: str) -> EncoderDecoderData:
        """Load or retrieve cached EncoderDecoderData."""
        if matrix_path not in self._encoder_cache:
            print(f"  [Adaptive] Загрузка матрицы: {os.path.basename(matrix_path)}")
            self._encoder_cache[matrix_path] = EncoderDecoderData(matrix_path)
        return self._encoder_cache[matrix_path]

    def _find_current_matrix_info(self, matrix_path: str) -> Optional[MatrixInfo]:
        """Find a MatrixInfo in the catalog matching the given path."""
        for m in self.catalog.matrices:
            if os.path.abspath(m.path) == os.path.abspath(matrix_path):
                return m
        return None

    def run_adaptive_sweep(self, encoder_decoder_data, settings, args,
                           encoding_method, ru_data=None):
        """Run SNR sweep with adaptive parameter selection between points.

        After each SNR point:
        1. Evaluate metrics via strategy
        2. Apply parameter changes if recommended
        3. Continue to next SNR point
        """
        # Import process_block from main to reuse
        from main import process_block, _process_batch_results

        start_time = time.time()
        num_steps = int(math.ceil((args.end_snr - args.initial_snr) / args.step_snr)) + 1

        # Cache initial encoder data
        self._encoder_cache[args.matrix] = encoder_decoder_data

        # Build initial adaptive state
        state = AdaptiveState(
            current_matrix_path=args.matrix,
            current_rate=encoder_decoder_data._rate,
            current_modulation=args.modulation,
            current_max_iterations=args.iterations,
            current_interleaver=args.interleaver,
            current_encoding_method=args.encoding_method,
        )

        current_encoder_data = encoder_decoder_data
        current_settings = settings
        current_encoding_method = encoding_method
        current_ru_data = ru_data

        snr_points = []
        adaptation_log = []

        print("Обработка блоков для различных значений SNR (адаптивный режим)...")
        print("-" * 60)

        for step in range(num_steps):
            current_snr = args.initial_snr + step * args.step_snr
            if current_snr > args.end_snr:
                current_snr = args.end_snr

            print(f"\nSNR: {current_snr:.2f} дБ  [rate={state.current_rate:.3f}, "
                  f"mod={'BPSK' if state.current_modulation == 1 else 'QPSK'}, "
                  f"iters={state.current_max_iterations}, interleaver={state.current_interleaver}]")
            print("-" * 60)

            # Log current state
            adaptation_log.append({
                'snr_db': current_snr,
                'matrix_path': state.current_matrix_path,
                'rate': state.current_rate,
                'modulation': state.current_modulation,
                'max_iterations': state.current_max_iterations,
                'interleaver': state.current_interleaver,
                'encoding_method': state.current_encoding_method,
            })

            channel_params = {
                'speed': args.speed,
                'snr': current_snr,
                'interference_snr': args.interference_snr,
                'mode': args.mode,
                'p': args.p,
                'modulation': state.current_modulation
            }

            # Run blocks for this SNR point
            total_fer = 0.0
            total_normalized_llr = 0.0
            successful_blocks = 0
            failed_blocks = 0
            error_bits_ber = 0
            total_convergence_iters = 0
            convergence_count = 0

            if args.threads > 1:
                # Pre-init decoder structures for multiprocessing
                if not hasattr(current_encoder_data, '_decoder_structures_initialized') or \
                   not current_encoder_data._decoder_structures_initialized:
                    current_encoder_data._init_decoder_structures()

                processed_count = 0
                with ProcessPoolExecutor(max_workers=args.threads) as executor:
                    futures = {
                        executor.submit(
                            process_block, block_num, current_encoder_data,
                            current_settings, current_encoding_method,
                            channel_params, args.ber, args.normalized_llr,
                            current_ru_data
                        ): block_num
                        for block_num in range(args.blocks)
                    }

                    batch_size = max(50, min(100, args.blocks // 10))
                    batch_results = []

                    for future in as_completed(futures):
                        try:
                            result_data = future.result()
                            batch_results.append(result_data)

                            if len(batch_results) >= batch_size:
                                (successful_blocks, failed_blocks, error_bits_ber,
                                 total_fer, total_normalized_llr,
                                 total_convergence_iters,
                                 convergence_count) = _process_batch_results(
                                    batch_results, args, successful_blocks,
                                    failed_blocks, error_bits_ber, total_fer,
                                    total_normalized_llr,
                                    total_convergence_iters, convergence_count)
                                processed_count += len(batch_results)
                                if processed_count % 10 == 0 or processed_count == args.blocks:
                                    print(f"  Обработано блоков: {processed_count}/{args.blocks}", end='\r')
                                batch_results = []
                        except Exception as e:
                            print(f"\nОшибка при обработке блока {futures[future]}: {e}")
                            failed_blocks += 1
                            processed_count += 1

                    if batch_results:
                        (successful_blocks, failed_blocks, error_bits_ber,
                         total_fer, total_normalized_llr,
                         total_convergence_iters,
                         convergence_count) = _process_batch_results(
                            batch_results, args, successful_blocks,
                            failed_blocks, error_bits_ber, total_fer,
                            total_normalized_llr,
                            total_convergence_iters, convergence_count)
                        processed_count += len(batch_results)

                print()
            else:
                # Single-threaded
                from data_buffer import DataBuffer

                channel_obj = Channel.create_channel(
                    args.speed, current_snr,
                    args.interference_snr if args.mode != 1 else 0.0,
                    args.mode, args.p, state.current_modulation
                )
                decoder = SPA_Decoder(current_encoder_data, current_settings)

                for block_num in range(args.blocks):
                    data_buffer = DataBuffer(current_encoder_data._k)

                    if current_encoding_method == EncodingMethod.RICHARDSON_URBANKE:
                        local_ru = current_ru_data if current_ru_data else \
                            current_encoder_data.prepare_richardson_urbanke_encoding()
                        data_buffer.encode_richardson_urbanke(local_ru)
                    else:
                        data_buffer.encode(current_encoder_data._g_transpose)

                    if current_settings.get_interleaver_type() != InterleaverType.NONE:
                        data_buffer.interleave(current_settings.get_interleaver_type())

                    channel_obj.process(data_buffer)

                    if current_settings.get_interleaver_type() != InterleaverType.NONE:
                        data_buffer.deinterleave(current_settings.get_interleaver_type())

                    result = decoder.decode(data_buffer)

                    if result == Result.OK:
                        successful_blocks += 1
                    else:
                        failed_blocks += 1

                    if args.fer and result != Result.OK:
                        total_fer += 1.0

                    if args.ber and result != Result.OK:
                        original_info = data_buffer._data[:current_encoder_data._k]
                        decoded_info = data_buffer._decoded_data[:current_encoder_data._k]
                        for i_index in range(len(original_info)):
                            i_value = decoded_info[i_index] ^ 1
                            if original_info[i_index] != i_value:
                                error_bits_ber += 1

                    if args.normalized_llr and decoder._normalized_llr_by_iterations:
                        total_normalized_llr += decoder._d_summarize_normalized_llr

                    conv_iter = decoder.convergence_iteration
                    if conv_iter >= 0:
                        total_convergence_iters += conv_iter
                        convergence_count += 1

                    if (block_num + 1) % 10 == 0 or (block_num + 1) == args.blocks:
                        print(f"  Обработано блоков: {block_num + 1}/{args.blocks}", end='\r')

                print()

            # Compute averages
            avg_normalized_llr = total_normalized_llr / args.blocks if args.normalized_llr else 0.0
            avg_fer = total_fer / args.blocks if args.fer else 0.0
            total_bits = current_encoder_data._k * args.blocks
            avg_ber = float(error_bits_ber) / total_bits if (args.ber and total_bits > 0) else 0.0
            avg_convergence = (total_convergence_iters / convergence_count
                               if convergence_count > 0 else 0.0)

            if args.normalized_llr:
                print(f"  Нормализованный LLR: {avg_normalized_llr:.6f}")
            if args.fer:
                print(f"  FER: {avg_fer:.6f}")
            if args.ber:
                print(f"  BER: {avg_ber:.6f}")
            print(f"  Успешно декодировано: {successful_blocks}/{args.blocks} "
                  f"({100.0 * successful_blocks / args.blocks:.2f}%)")

            snr_point = SNRPointResult(
                snr_db=current_snr,
                ber=avg_ber,
                fer=avg_fer,
                avg_normalized_llr=avg_normalized_llr,
                total_blocks=args.blocks,
                successful_blocks=successful_blocks,
                failed_blocks=failed_blocks,
                avg_convergence_iterations=avg_convergence,
                matrix_path=state.current_matrix_path,
                modulation=state.current_modulation,
                max_iterations=state.current_max_iterations,
                interleaver=state.current_interleaver,
                encoding_method=state.current_encoding_method,
            )
            snr_points.append(snr_point)

            # Evaluate and adapt for next SNR point
            action = self.strategy.evaluate(state, snr_point)
            if action:
                print(f"  [Adaptive] {action.reason}")
                self._apply_action(action, state, current_encoder_data, current_settings, args)
                current_encoder_data = self._encoder_cache.get(
                    state.current_matrix_path, current_encoder_data
                )
                current_settings = self._build_settings(state, args)
                # Reset RU data if matrix changed
                if action.new_matrix_path:
                    current_ru_data = None
                    if state.current_encoding_method == 'richardson-urbanke':
                        current_ru_data = current_encoder_data.prepare_richardson_urbanke_encoding()

        # Print summary
        print()
        print("=" * 60)
        print("Итоговые результаты по SNR (адаптивный режим):")
        print("=" * 60)

        if args.ber:
            print("\nSNR -> BER (rate):")
            for sp in snr_points:
                rate_info = f" [rate from {os.path.basename(sp.matrix_path)}]"
                print(f"  {sp.snr_db:.2f} дБ -> {sp.ber:.6f}{rate_info}")

        if args.fer:
            print("\nSNR -> FER:")
            for sp in snr_points:
                print(f"  {sp.snr_db:.2f} дБ -> {sp.fer:.6f}")

        print("=" * 60)
        print("Обработка завершена успешно!")

        elapsed = time.time() - start_time

        config = SimulationConfig(
            matrix_path=args.matrix,
            n=encoder_decoder_data._n,
            m=encoder_decoder_data._m,
            k=encoder_decoder_data._k,
            rate=encoder_decoder_data._rate,
            blocks=args.blocks,
            max_iterations=args.iterations,
            encoding_method=args.encoding_method,
            interleaver_type=args.interleaver,
            decoder_type=args.decoder,
            channel_mode=args.mode,
            modulation=args.modulation,
            speed=args.speed,
            snr_range=(args.initial_snr, args.end_snr, args.step_snr),
            threads=args.threads,
            timestamp=datetime.now().isoformat(),
            interference_snr=args.interference_snr,
            p=args.p,
        )

        return SimulationResult(
            config=config,
            snr_points=snr_points,
            wall_clock_seconds=elapsed,
            adaptation_log=adaptation_log,
        )

    def _apply_action(self, action: AdaptiveAction, state: AdaptiveState,
                      current_encoder_data: EncoderDecoderData,
                      current_settings: Settings, args):
        """Apply an adaptive action by modifying state."""
        current_info = self._find_current_matrix_info(state.current_matrix_path)

        if action.new_matrix_path == "__LOWER_RATE__" and current_info:
            lower = self.catalog.get_lower_rate(current_info)
            if lower:
                state.current_matrix_path = lower.path
                state.current_rate = lower.rate
                self._get_encoder_decoder_data(lower.path)
                print(f"  [Adaptive] Матрица: {lower.name} (rate={lower.rate:.3f})")
        elif action.new_matrix_path == "__HIGHER_RATE__" and current_info:
            higher = self.catalog.get_higher_rate(current_info)
            if higher:
                state.current_matrix_path = higher.path
                state.current_rate = higher.rate
                self._get_encoder_decoder_data(higher.path)
                print(f"  [Adaptive] Матрица: {higher.name} (rate={higher.rate:.3f})")

        if action.new_max_iterations is not None:
            state.current_max_iterations = action.new_max_iterations

        if action.new_modulation is not None:
            state.current_modulation = action.new_modulation

        if action.new_interleaver is not None:
            state.current_interleaver = action.new_interleaver

    def _build_settings(self, state: AdaptiveState, args) -> Settings:
        """Build a Settings object from current adaptive state."""
        s = Settings()
        s.set_blocks_cnt(args.blocks)
        s.set_max_iterations(state.current_max_iterations)

        interleaver_map = {
            'none': InterleaverType.NONE,
            'regular': InterleaverType.REGULAR,
            'random': InterleaverType.RANDOM,
            'srandom': InterleaverType.SRANDOM,
        }
        s.set_interleaver_type(interleaver_map.get(state.current_interleaver, InterleaverType.NONE))
        if state.current_interleaver == 'srandom':
            s.set_s_param(args.s_param)

        s.set_decoder_type(LDPCDecoderType.SUM_PRODUCT)
        s.set_ber_calculate(args.ber)
        s.set_fer_calculate(args.fer)
        s.set_normalized_llr_calculate(args.normalized_llr)

        return s
