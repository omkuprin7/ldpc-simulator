"""Integration tests: small end-to-end simulation runs."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.integration
class TestEndToEnd:
    def test_basic_simulation(self, bch_matrix_path):
        """Run a minimal simulation and verify structured results."""
        if not os.path.exists(bch_matrix_path):
            pytest.skip("BCH matrix file not found")

        from encoder_decoder_data import EncoderDecoderData
        from settings import Settings
        from enums import InterleaverType, LDPCDecoderType, EncodingMethod

        encoder_decoder_data = EncoderDecoderData(bch_matrix_path)
        assert encoder_decoder_data._n == 7
        assert encoder_decoder_data._k == 4

        settings = Settings()
        settings.set_blocks_cnt(10)
        settings.set_max_iterations(5)
        settings.set_interleaver_type(InterleaverType.NONE)
        settings.set_decoder_type(LDPCDecoderType.SUM_PRODUCT)
        settings.set_ber_calculate(True)
        settings.set_fer_calculate(True)
        settings.set_normalized_llr_calculate(True)

        # Create a mock args namespace
        class Args:
            matrix = bch_matrix_path
            blocks = 10
            iterations = 5
            interleaver = 'none'
            decoder = 'sumproduct'
            speed = 1.0
            initial_snr = 0.0
            end_snr = 2.0
            step_snr = 1.0
            interference_snr = 1.0
            mode = 1
            p = 0.1
            modulation = 1
            s_param = 2
            ber = True
            fer = True
            normalized_llr = True
            encoding_method = 'standard'
            ru_gap = None
            threads = 1

        args = Args()

        from main import run_simulation
        result = run_simulation(
            encoder_decoder_data, settings, args, EncodingMethod.STANDARD
        )

        assert result is not None
        assert len(result.snr_points) == 3  # 0.0, 1.0, 2.0
        assert result.config.n == 7
        assert result.config.k == 4
        assert result.wall_clock_seconds > 0

        for sp in result.snr_points:
            assert sp.total_blocks == 10
            assert sp.successful_blocks + sp.failed_blocks == 10
            assert 0.0 <= sp.ber <= 1.0
            assert 0.0 <= sp.fer <= 1.0

    def test_json_roundtrip_from_simulation(self, bch_matrix_path):
        """Run simulation, export to JSON, reload, verify."""
        if not os.path.exists(bch_matrix_path):
            pytest.skip("BCH matrix file not found")

        from encoder_decoder_data import EncoderDecoderData
        from settings import Settings
        from enums import InterleaverType, LDPCDecoderType, EncodingMethod
        from results import SimulationResult

        encoder_decoder_data = EncoderDecoderData(bch_matrix_path)
        settings = Settings()
        settings.set_blocks_cnt(5)
        settings.set_max_iterations(3)
        settings.set_interleaver_type(InterleaverType.NONE)
        settings.set_decoder_type(LDPCDecoderType.SUM_PRODUCT)
        settings.set_ber_calculate(True)
        settings.set_fer_calculate(True)
        settings.set_normalized_llr_calculate(False)

        class Args:
            matrix = bch_matrix_path
            blocks = 5
            iterations = 3
            interleaver = 'none'
            decoder = 'sumproduct'
            speed = 1.0
            initial_snr = 0.0
            end_snr = 1.0
            step_snr = 1.0
            interference_snr = 1.0
            mode = 1
            p = 0.1
            modulation = 1
            s_param = 2
            ber = True
            fer = True
            normalized_llr = False
            encoding_method = 'standard'
            ru_gap = None
            threads = 1

        from main import run_simulation
        result = run_simulation(
            encoder_decoder_data, settings, Args(), EncodingMethod.STANDARD
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            result.to_json(path)
            loaded = SimulationResult.from_json(path)
            assert len(loaded.snr_points) == len(result.snr_points)
            assert loaded.config.n == result.config.n
        finally:
            os.unlink(path)

    def test_convergence_iteration_tracked(self, bch_matrix_path):
        """Verify that convergence_iteration is correctly tracked."""
        if not os.path.exists(bch_matrix_path):
            pytest.skip("BCH matrix file not found")

        from encoder_decoder_data import EncoderDecoderData
        from spa_decoder import SPA_Decoder
        from data_buffer import DataBuffer
        from channel import Channel
        from settings import Settings
        from enums import InterleaverType, LDPCDecoderType, Result

        encoder_decoder_data = EncoderDecoderData(bch_matrix_path)
        settings = Settings()
        settings.set_max_iterations(10)
        settings.set_decoder_type(LDPCDecoderType.SUM_PRODUCT)
        settings.set_normalized_llr_calculate(False)

        # Use high SNR for likely successful decoding
        channel = Channel.create_channel(1.0, 10.0, 0.0, 1, 0.1, 1)

        decoder = SPA_Decoder(encoder_decoder_data, settings)
        data_buffer = DataBuffer(encoder_decoder_data._k)
        data_buffer.encode(encoder_decoder_data._g_transpose)
        channel.process(data_buffer)
        result = decoder.decode(data_buffer)

        if result == Result.OK:
            assert decoder.convergence_iteration >= 0
            assert decoder.convergence_iteration < 10
        else:
            assert decoder.convergence_iteration == -1
