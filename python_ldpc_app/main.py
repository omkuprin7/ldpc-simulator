#!/usr/bin/env python3
"""
LDPC Application - Console Version
Консольное приложение для работы с LDPC кодами
"""
import sys
import argparse
import os
import math
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_buffer import DataBuffer
from encoder_decoder_data import EncoderDecoderData
from channel import Channel
from spa_decoder import SPA_Decoder
from settings import Settings
from enums import InterleaverType, LDPCDecoderType, Result, EncodingMethod
from results import SimulationResult, SimulationConfig, SNRPointResult



def calculate_ber(original_data, decoded_data):
    """Calculate Bit Error Rate."""
    if len(original_data) != len(decoded_data):
        return 1.0

    errors = sum(1 for i in range(len(original_data)) if original_data[i] != decoded_data[i])
    return float(errors) / len(original_data) if len(original_data) > 0 else 0.0


def calculate_fer(decoding_result):
    """Calculate Frame Error Rate.
    FER = 1.0 if decoding failed (result != OK), else 0.0
    """
    return 1.0 if decoding_result != Result.OK else 0.0


def process_block(block_num, encoder_decoder_data, settings, encoding_method,
                  channel_params, calculate_ber=False, calculate_normalized_llr=False, ru_data=None):
    """Process a single block: encode, transmit through channel, decode.
    This function is designed to be process-safe and can run in parallel using ProcessPoolExecutor.

    IMPORTANT: Architecture for multiprocessing:
    - encoder_decoder_data: Shared read-only data, passed once to each process via pickle
      (scipy.sparse matrices are efficiently pickle-able)
    - decoder: Created separately in EACH process (not shared)
    - channel: Created separately in EACH process (not shared)
    - data_buffer: Created separately in EACH process (not shared)

    Note: All parameters must be pickle-able for ProcessPoolExecutor.
    - encoder_decoder_data: Contains scipy.sparse matrices (pickle-able, read-only)
    - settings: Simple class (pickle-able, read-only)
    - encoding_method: Enum (pickle-able)
    - channel_params: Dictionary (pickle-able)
    - ru_data: Dictionary (pickle-able, read-only)

    Args:
        block_num: Block number
        encoder_decoder_data: Encoder/decoder data (shared read-only, one instance per process via pickle)
        settings: Settings (shared read-only, one instance per process via pickle)
        encoding_method: Encoding method enum
        channel_params: Channel parameters dictionary
        calculate_ber: Boolean flag for BER calculation (instead of passing args)
        calculate_normalized_llr: Boolean flag for normalized LLR calculation (instead of passing args)
        ru_data: Pre-computed Richardson-Urbanke data (optional, to avoid repeated calls)

    Returns:
        tuple: (block_num, result, error_bits_ber, normalized_llr, convergence_iteration)
    """
    # Create decoder instance for THIS process
    # Each process gets its own decoder instance (not shared)
    # encoder_decoder_data is read-only and shared via pickle serialization
    decoder = SPA_Decoder(encoder_decoder_data, settings)

    # Create channel with same parameters
    if channel_params['mode'] == 1:
        channel = Channel.create_channel(
            channel_params['speed'],
            channel_params['snr'],
            0.0,
            channel_params['mode'],
            channel_params['p'],
            channel_params['modulation']
        )
    else:
        channel = Channel.create_channel(
            channel_params['speed'],
            channel_params['snr'],
            channel_params['interference_snr'],
            channel_params['mode'],
            channel_params['p'],
            channel_params['modulation']
        )

    # Create data buffer
    data_buffer = DataBuffer(encoder_decoder_data._k)

    # Encoding
    if encoding_method == EncodingMethod.RICHARDSON_URBANKE:
        # Use pre-computed ru_data if provided, otherwise get it (cached anyway)
        if ru_data is None:
            ru_data = encoder_decoder_data.prepare_richardson_urbanke_encoding()
        data_buffer.encode_richardson_urbanke(ru_data)
    else:
        data_buffer.encode(encoder_decoder_data._g_transpose)

    # Interleaving
    if settings.get_interleaver_type() != InterleaverType.NONE:
        data_buffer.interleave(settings.get_interleaver_type())

    # Channel transmission
    channel.process(data_buffer)

    # Deinterleaving
    if settings.get_interleaver_type() != InterleaverType.NONE:
        data_buffer.deinterleave(settings.get_interleaver_type())

    # Decoding
    result = decoder.decode(data_buffer)

    # Calculate metrics
    error_bits_ber = 0
    normalized_llr = 0.0

    if calculate_ber:
        original_info = data_buffer._data[:encoder_decoder_data._k]
        decoded_info = data_buffer._decoded_data[:encoder_decoder_data._k]

        if result != Result.OK:
            for i_index in range(len(original_info)):
                i_value = decoded_info[i_index] ^ 1
                if original_info[i_index] != i_value:
                    error_bits_ber += 1

    if calculate_normalized_llr:
        if decoder._normalized_llr_by_iterations:
            normalized_llr = decoder._d_summarize_normalized_llr

    convergence_iteration = decoder.convergence_iteration

    return (block_num, result, error_bits_ber, normalized_llr, convergence_iteration)


def _process_batch_results(batch_results, args, successful_blocks, failed_blocks,
                           error_bits_ber, total_fer, total_normalized_llr,
                           total_convergence_iters, convergence_count):
    """Process a batch of block results, updating accumulators in-place.
    Returns updated values as a tuple."""
    for block_num, result, block_error_bits, block_normalized_llr, conv_iter in batch_results:
        if result == Result.OK:
            successful_blocks += 1
        else:
            failed_blocks += 1

        if args.fer:
            if result != Result.OK:
                total_fer += 1.0

        if args.ber:
            error_bits_ber += block_error_bits

        if args.normalized_llr:
            total_normalized_llr += block_normalized_llr

        if conv_iter >= 0:
            total_convergence_iters += conv_iter
            convergence_count += 1

    return (successful_blocks, failed_blocks, error_bits_ber, total_fer,
            total_normalized_llr, total_convergence_iters, convergence_count)


def run_simulation(encoder_decoder_data, settings, args, encoding_method, ru_data=None):
    """Run the full SNR sweep simulation and return structured results.

    Args:
        encoder_decoder_data: Loaded encoder/decoder data
        settings: Simulation settings
        args: Parsed CLI arguments
        encoding_method: EncodingMethod enum
        ru_data: Pre-computed Richardson-Urbanke data (or None)

    Returns:
        SimulationResult with all SNR point data
    """
    start_time = time.time()

    num_steps = int(math.ceil((args.end_snr - args.initial_snr) / args.step_snr)) + 1

    # Legacy lists for backward-compatible console output
    snr_to_ber = []
    snr_to_fer = []
    snr_to_normalized_llr = []

    # Structured results
    snr_points = []

    print("Обработка блоков для различных значений SNR...")
    print("-" * 60)

    for step in range(num_steps):
        current_snr = args.initial_snr + step * args.step_snr
        if current_snr > args.end_snr:
            current_snr = args.end_snr

        print(f"\nSNR: {current_snr:.2f} дБ")
        print("-" * 60)

        # Create channel for current SNR
        if args.mode == 1:
            channel = Channel.create_channel(args.speed, current_snr, 0.0, args.mode, args.p, args.modulation)
        else:
            channel = Channel.create_channel(args.speed, current_snr, args.interference_snr, args.mode, args.p, args.modulation)

        # Create decoder
        decoder = SPA_Decoder(encoder_decoder_data, settings)

        # Statistics for current SNR
        total_fer = 0.0
        total_normalized_llr = 0.0
        successful_blocks = 0
        failed_blocks = 0
        error_bits_ber = 0
        total_convergence_iters = 0
        convergence_count = 0

        channel_params = {
            'speed': args.speed,
            'snr': current_snr,
            'interference_snr': args.interference_snr,
            'mode': args.mode,
            'p': args.p,
            'modulation': args.modulation
        }

        if args.threads > 1:
            # Multi-process block processing
            processed_count = 0
            local_ru_data = None
            if encoding_method == EncodingMethod.RICHARDSON_URBANKE:
                local_ru_data = ru_data if ru_data else encoder_decoder_data.prepare_richardson_urbanke_encoding()

            with ProcessPoolExecutor(max_workers=args.threads) as executor:
                futures = {
                    executor.submit(
                        process_block, block_num, encoder_decoder_data, settings,
                        encoding_method, channel_params, args.ber, args.normalized_llr,
                        local_ru_data
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
                            (successful_blocks, failed_blocks, error_bits_ber, total_fer,
                             total_normalized_llr, total_convergence_iters,
                             convergence_count) = _process_batch_results(
                                batch_results, args, successful_blocks, failed_blocks,
                                error_bits_ber, total_fer, total_normalized_llr,
                                total_convergence_iters, convergence_count)
                            processed_count += len(batch_results)
                            if processed_count % 10 == 0 or processed_count == args.blocks:
                                print(f"  Обработано блоков: {processed_count}/{args.blocks}", end='\r')
                            batch_results = []
                    except Exception as e:
                        print(f"\nОшибка при обработке блока {futures[future]}: {e}")
                        failed_blocks += 1
                        processed_count += 1

                # Process remaining batch
                if batch_results:
                    (successful_blocks, failed_blocks, error_bits_ber, total_fer,
                     total_normalized_llr, total_convergence_iters,
                     convergence_count) = _process_batch_results(
                        batch_results, args, successful_blocks, failed_blocks,
                        error_bits_ber, total_fer, total_normalized_llr,
                        total_convergence_iters, convergence_count)
                    processed_count += len(batch_results)

            print()
        else:
            # Single-threaded processing
            for block_num in range(args.blocks):
                data_buffer = DataBuffer(encoder_decoder_data._k)

                if encoding_method == EncodingMethod.RICHARDSON_URBANKE:
                    local_ru_data = ru_data if ru_data else encoder_decoder_data.prepare_richardson_urbanke_encoding()
                    data_buffer.encode_richardson_urbanke(local_ru_data)
                else:
                    data_buffer.encode(encoder_decoder_data._g_transpose)

                if settings.get_interleaver_type() != InterleaverType.NONE:
                    data_buffer.interleave(settings.get_interleaver_type())

                channel.process(data_buffer)

                if settings.get_interleaver_type() != InterleaverType.NONE:
                    data_buffer.deinterleave(settings.get_interleaver_type())

                result = decoder.decode(data_buffer)

                if result == Result.OK:
                    successful_blocks += 1
                else:
                    failed_blocks += 1

                if args.fer:
                    if result != Result.OK:
                        total_fer += 1.0

                if args.ber:
                    original_info = data_buffer._data[:encoder_decoder_data._k]
                    decoded_info = data_buffer._decoded_data[:encoder_decoder_data._k]
                    if result != Result.OK:
                        for i_index in range(len(original_info)):
                            i_value = decoded_info[i_index] ^ 1
                            if original_info[i_index] != i_value:
                                error_bits_ber += 1

                if args.normalized_llr:
                    if decoder._normalized_llr_by_iterations:
                        total_normalized_llr += decoder._d_summarize_normalized_llr

                conv_iter = decoder.convergence_iteration
                if conv_iter >= 0:
                    total_convergence_iters += conv_iter
                    convergence_count += 1

                if (block_num + 1) % 10 == 0 or (block_num + 1) == args.blocks:
                    print(f"  Обработано блоков: {block_num + 1}/{args.blocks}", end='\r')

            print()

        # Compute averages for current SNR
        avg_normalized_llr = 0.0
        avg_fer = 0.0
        avg_ber = 0.0
        avg_convergence = 0.0

        if args.normalized_llr:
            avg_normalized_llr = total_normalized_llr / args.blocks
            snr_to_normalized_llr.append((current_snr, avg_normalized_llr))
            print(f"  Нормализованный LLR: {avg_normalized_llr:.6f}")

        if args.fer:
            avg_fer = total_fer / args.blocks
            snr_to_fer.append((current_snr, avg_fer))
            print(f"  FER: {avg_fer:.6f}")

        if args.ber:
            total_bits = encoder_decoder_data._k * args.blocks
            avg_ber = float(error_bits_ber) / total_bits if total_bits > 0 else 0.0
            snr_to_ber.append((current_snr, avg_ber))
            print(f"  BER: {avg_ber:.6f}")

        if convergence_count > 0:
            avg_convergence = total_convergence_iters / convergence_count

        print(f"  Успешно декодировано: {successful_blocks}/{args.blocks} ({100.0 * successful_blocks / args.blocks:.2f}%)")

        # Build structured SNR point result
        snr_point = SNRPointResult(
            snr_db=current_snr,
            ber=avg_ber,
            fer=avg_fer,
            avg_normalized_llr=avg_normalized_llr,
            total_blocks=args.blocks,
            successful_blocks=successful_blocks,
            failed_blocks=failed_blocks,
            avg_convergence_iterations=avg_convergence,
            matrix_path=args.matrix,
            modulation=args.modulation,
            max_iterations=args.iterations,
            interleaver=args.interleaver,
            encoding_method=args.encoding_method,
        )
        snr_points.append(snr_point)

    # Print summary (backward compatible)
    print()
    print("=" * 60)
    print("Итоговые результаты по SNR:")
    print("=" * 60)

    if args.ber:
        print("\nSNR -> BER:")
        for snr, ber in snr_to_ber:
            print(f"  {snr:.2f} дБ -> {ber:.6f}")

    if args.fer:
        print("\nSNR -> FER:")
        for snr, fer in snr_to_fer:
            print(f"  {snr:.2f} дБ -> {fer:.6f}")

    if args.normalized_llr:
        print("\nSNR -> Normalized LLR:")
        for snr, llr in snr_to_normalized_llr:
            print(f"  {snr:.2f} дБ -> {llr:.6f}")

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
    )


def main():
    parser = argparse.ArgumentParser(
        description='LDPC Application - Консольное приложение для работы с LDPC кодами',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --matrix ../Channel_Codes_Database/BCH_7_4_1_strip.alist.txt --blocks 100
  python main.py --matrix ../Channel_Codes_Database/BCH_7_4_1_strip.alist.txt --blocks 50 --iterations 10 --interleaver random
        """
    )

    parser.add_argument('--matrix', '-m', type=str, required=True,
                        help='Путь к файлу матрицы проверки четности (parity check matrix)')
    parser.add_argument('--blocks', '-b', type=int, default=100,
                        help='Количество блоков для обработки (по умолчанию: 100)')
    parser.add_argument('--iterations', '-i', type=int, default=5,
                        help='Максимальное количество итераций декодера (по умолчанию: 5)')
    parser.add_argument('--interleaver', '-il', type=str, choices=['none', 'regular', 'random', 'srandom'],
                        default='none', help='Тип интерливера (по умолчанию: none)')
    parser.add_argument('--decoder', '-d', type=str, choices=['bitflipping', 'sumproduct'],
                        default='sumproduct', help='Тип декодера (по умолчанию: sumproduct)')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                        help='Скорость передачи (по умолчанию: 1.0)')
    parser.add_argument('--initial-snr', type=float, default=0.0,
                        help='Начальное SNR в дБ (по умолчанию: 0.0)')
    parser.add_argument('--end-snr', type=float, default=5.0,
                        help='Конечное SNR в дБ (по умолчанию: 5.0)')
    parser.add_argument('--step-snr', type=float, default=0.5,
                        help='Шаг SNR в дБ (по умолчанию: 0.5)')
    parser.add_argument('--interference-snr', type=float, default=1.0,
                        help='SNR помехи в дБ для режимов 2 и 3 (по умолчанию: 1.0)')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], default=1,
                        help='Режим канала: 1=АБГШ, 2=АБГШ+помеха в части полосы, 3=АБГШ+встречная помеха (по умолчанию: 1)')
    parser.add_argument('--p', type=float, default=0.1,
                        help='Параметр p (gamma) для режимов 2 и 3 (по умолчанию: 0.1)')
    parser.add_argument('--modulation', '-mod', type=int, choices=[1, 2], default=1,
                        help='Тип модуляции: 1=ФМ-2, 2=ФМ-4 (по умолчанию: 1)')
    parser.add_argument('--s-param', type=int, default=2,
                        help='Параметр S для S-Random интерливера (по умолчанию: 2)')
    parser.add_argument('--ber', action='store_true',
                        help='Вычислять BER (Bit Error Rate)')
    parser.add_argument('--fer', action='store_true',
                        help='Вычислять FER (Frame Error Rate)')
    parser.add_argument('--normalized-llr', action='store_true',
                        help='Вычислять нормализованный LLR')
    parser.add_argument('--encoding-method', '-e', type=str,
                        choices=['standard', 'richardson-urbanke'],
                        default='standard',
                        help='Метод кодирования: standard (стандартный) или richardson-urbanke (Ричардсон-Урбанке) (по умолчанию: standard)')
    parser.add_argument('--ru-gap', type=int, default=None,
                        help='Gap для метода Ричардсона-Урбанке (по умолчанию: автоматический поиск минимального gap)')
    parser.add_argument('--threads', '-t', type=int, default=1,
                        help='Количество потоков для параллельной обработки блоков (по умолчанию: 1, без многопоточности)')

    # Export flags
    parser.add_argument('--output-json', type=str, default=None,
                        help='Экспорт результатов в JSON файл')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Экспорт результатов в CSV файл')

    # Visualization flags
    parser.add_argument('--plot', action='store_true',
                        help='Показать графики после симуляции')
    parser.add_argument('--plot-save', type=str, default=None,
                        help='Сохранить графики в указанную директорию')

    # Adaptive mode flags
    parser.add_argument('--adaptive', action='store_true',
                        help='Включить адаптивный подбор параметров')
    parser.add_argument('--adaptive-strategy', type=str, choices=['threshold'],
                        default='threshold',
                        help='Стратегия адаптивного подбора (по умолчанию: threshold)')
    parser.add_argument('--matrix-dir', type=str, default=None,
                        help='Путь к директории с матрицами (по умолчанию: ../Channel_Codes_Database)')
    parser.add_argument('--adaptive-high-ber', type=float, default=1e-2,
                        help='Порог BER для переключения на более защищённый код (по умолчанию: 1e-2)')
    parser.add_argument('--adaptive-low-ber', type=float, default=1e-5,
                        help='Порог BER для переключения на более быстрый код (по умолчанию: 1e-5)')

    args = parser.parse_args()

    # Проверка существования файла матрицы
    if not os.path.exists(args.matrix):
        print(f"Ошибка: Файл матрицы не найден: {args.matrix}")
        sys.exit(1)

    print("=" * 60)
    print("LDPC Application - Консольное приложение")
    print("=" * 60)
    print(f"Файл матрицы: {args.matrix}")
    print(f"Количество блоков: {args.blocks}")
    print(f"Максимальные итерации: {args.iterations}")
    print(f"Тип интерливера: {args.interleaver}")
    print(f"Тип декодера: {args.decoder}")
    print(f"Метод кодирования: {args.encoding_method}")
    print(f"Режим канала: {args.mode}")
    print(f"Количество потоков: {args.threads}")
    print(f"Диапазон SNR: {args.initial_snr} - {args.end_snr} дБ (шаг: {args.step_snr} дБ)")
    if args.mode in [2, 3]:
        print(f"SNR помехи: {args.interference_snr} дБ")
    if args.adaptive:
        print(f"Адаптивный режим: включен (стратегия: {args.adaptive_strategy})")
    print("=" * 60)
    print()

    # Запоминаем время начала работы (перед try блоком)
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"Время начала работы: {start_datetime.strftime('%d.%m.%Y %H:%M:%S')}")
    print("-" * 60)
    print()

    try:
        # Загрузка данных кодера/декодера
        print("Загрузка матрицы проверки четности...")
        encoder_decoder_data = EncoderDecoderData(args.matrix)
        print(f"Параметры кода: n={encoder_decoder_data._n}, m={encoder_decoder_data._m}, k={encoder_decoder_data._k}, rate={encoder_decoder_data._rate:.4f}")

        # Pre-initialize decoder structures for multi-threading efficiency
        if args.threads > 1:
            print("Инициализация структур декодера для многопоточности...")
            encoder_decoder_data._init_decoder_structures()
            print("Готово.")

        print()

        # Подготовка метода кодирования
        encoding_method = EncodingMethod.STANDARD
        ru_data = None
        if args.encoding_method == 'richardson-urbanke':
            encoding_method = EncodingMethod.RICHARDSON_URBANKE
            print("Подготовка матрицы для метода Ричардсона-Урбанке...")
            if args.ru_gap is not None:
                print(f"Заданный gap: {args.ru_gap}")
            ru_data = encoder_decoder_data.prepare_richardson_urbanke_encoding(gap=args.ru_gap)
            gap = ru_data['gap']
            if args.ru_gap is not None:
                print(f"Используемый gap: {gap} (заданный: {args.ru_gap})")
            else:
                print(f"Используемый gap: {gap} (автоматически найденный минимальный)")
            if gap > 0:
                print(f"  Структура H_ru: [A B T; C D E]")
                print(f"  Размеры: A({ru_data['m']}-{gap} x {ru_data['k']}), B({ru_data['m']}-{gap} x {gap}), T({ru_data['m']}-{gap} x {ru_data['m']}-{gap})")
                print(f"            C({gap} x {ru_data['k']}), D({gap} x {gap}), E({gap} x {ru_data['m']}-{gap})")
            else:
                print(f"  Структура H_ru: [A | I_m] (стандартная форма, gap = 0)")
            print()

        # Настройки
        settings = Settings()
        settings.set_blocks_cnt(args.blocks)
        settings.set_max_iterations(args.iterations)

        # Установка типа интерливера
        if args.interleaver == 'regular':
            settings.set_interleaver_type(InterleaverType.REGULAR)
        elif args.interleaver == 'random':
            settings.set_interleaver_type(InterleaverType.RANDOM)
        elif args.interleaver == 'srandom':
            settings.set_interleaver_type(InterleaverType.SRANDOM)
            settings.set_s_param(args.s_param)
        else:
            settings.set_interleaver_type(InterleaverType.NONE)

        # Установка типа декодера
        if args.decoder == 'bitflipping':
            settings.set_decoder_type(LDPCDecoderType.BIT_FLIPPING)
        else:
            settings.set_decoder_type(LDPCDecoderType.SUM_PRODUCT)

        settings.set_ber_calculate(args.ber)
        settings.set_fer_calculate(args.fer)
        settings.set_normalized_llr_calculate(args.normalized_llr)

        # Run simulation
        if args.adaptive:
            from adaptive import AdaptiveController, ThresholdStrategy
            from matrix_catalog import MatrixCatalog

            matrix_dir = args.matrix_dir
            if matrix_dir is None:
                # Default: look relative to the matrix file or use ../Channel_Codes_Database
                matrix_dir = os.path.join(os.path.dirname(os.path.abspath(args.matrix)), '..')
                if not os.path.isdir(matrix_dir):
                    matrix_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Channel_Codes_Database')

            catalog = MatrixCatalog(matrix_dir)
            strategy = ThresholdStrategy(
                high_ber_threshold=args.adaptive_high_ber,
                low_ber_threshold=args.adaptive_low_ber,
            )
            controller = AdaptiveController(strategy, catalog)
            sim_result = controller.run_adaptive_sweep(
                encoder_decoder_data, settings, args, encoding_method, ru_data
            )
        else:
            sim_result = run_simulation(
                encoder_decoder_data, settings, args, encoding_method, ru_data
            )

        # Timing
        end_time = time.time()
        end_datetime = datetime.now()
        elapsed_time = end_time - start_time

        print()
        print("=" * 60)
        print("Информация о времени выполнения:")
        print("=" * 60)
        print(f"Время начала работы: {start_datetime.strftime('%d.%m.%Y %H:%M:%S')}")
        print(f"Время окончания работы: {end_datetime.strftime('%d.%m.%Y %H:%M:%S')}")

        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60

        if hours > 0:
            print(f"Время работы: {hours} ч {minutes} мин {seconds:.2f} сек ({elapsed_time:.2f} секунд)")
        elif minutes > 0:
            print(f"Время работы: {minutes} мин {seconds:.2f} сек ({elapsed_time:.2f} секунд)")
        else:
            print(f"Время работы: {seconds:.2f} сек")
        print("=" * 60)

        # Export results
        if args.output_json:
            sim_result.to_json(args.output_json)
            print(f"\nРезультаты экспортированы в JSON: {args.output_json}")

        if args.output_csv:
            sim_result.to_csv(args.output_csv)
            print(f"Результаты экспортированы в CSV: {args.output_csv}")

        # Visualization
        if args.plot or args.plot_save:
            try:
                from visualization import SimulationPlotter
                plotter = SimulationPlotter(sim_result)
                plotter.plot_combined_dashboard(save_dir=args.plot_save)
                if args.plot:
                    import matplotlib.pyplot as plt
                    plt.show()
            except ImportError:
                print("\nПредупреждение: matplotlib не установлен. Установите: pip install matplotlib")

    except Exception as e:
        # Вычисляем время работы даже при ошибке
        end_time = time.time()
        end_datetime = datetime.now()
        elapsed_time = end_time - start_time

        print()
        print("=" * 60)
        print("Ошибка при выполнении:")
        print("=" * 60)
        print(f"Ошибка: {e}")
        print()
        print(f"Время начала работы: {start_datetime.strftime('%d.%m.%Y %H:%M:%S')}")
        print(f"Время окончания работы: {end_datetime.strftime('%d.%m.%Y %H:%M:%S')}")
        print(f"Время работы до ошибки: {elapsed_time:.2f} секунд")
        print("=" * 60)

        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
