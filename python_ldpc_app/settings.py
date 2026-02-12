from enums import InterleaverType, LDPCDecoderType


class Settings:
    def __init__(self):
        self._i_blocks_cnt = 100
        self._max_iterations = 5
        self._interleaver_type = InterleaverType.NONE
        self._decoder_type = LDPCDecoderType.BIT_FLIPPING
        self._b_ber_calculate = True
        self._b_fer_calculate = False
        self._b_is_calculate_normalized_llr = False
        self._d_s_param = -1

    def set_blocks_cnt(self, i_num_blocks):
        self._i_blocks_cnt = i_num_blocks

    def get_blocks_cnt(self):
        return self._i_blocks_cnt

    def set_max_iterations(self, i_max_iter):
        self._max_iterations = i_max_iter

    def get_max_iterations(self):
        return self._max_iterations

    def set_interleaver_type(self, e_int_type):
        self._interleaver_type = e_int_type

    def get_interleaver_type(self):
        return self._interleaver_type

    def set_decoder_type(self, e_decoder_type):
        self._decoder_type = e_decoder_type

    def get_decoder_type(self):
        return self._decoder_type

    def print(self):
        print(f"Block count: {self._i_blocks_cnt}")
        print("Interleaver type: ", end="")
        if self._interleaver_type == InterleaverType.REGULAR:
            print("Regular;")
        elif self._interleaver_type == InterleaverType.RANDOM:
            print("Random;")
        elif self._interleaver_type == InterleaverType.SRANDOM:
            print("S-Random;")
        else:
            print("None;")

        print("Decoder type: ", end="")
        if self._decoder_type == LDPCDecoderType.BIT_FLIPPING:
            print("Bit-flipped algorithm;")
        elif self._decoder_type == LDPCDecoderType.SUM_PRODUCT:
            print("Sum-product algorithm;")

    def set_ber_calculate(self, b_is_need_ber_calculate):
        self._b_ber_calculate = b_is_need_ber_calculate

    def is_ber_calculate(self):
        return self._b_ber_calculate

    def set_fer_calculate(self, b_is_need_fer_calculate):
        self._b_fer_calculate = b_is_need_fer_calculate

    def is_fer_calculate(self):
        return self._b_fer_calculate

    def set_normalized_llr_calculate(self, b_is_need_normalized_llr_calculate):
        self._b_is_calculate_normalized_llr = b_is_need_normalized_llr_calculate

    def is_normalized_llr_calculate(self):
        return self._b_is_calculate_normalized_llr

    def get_interleaver_type_name(self):
        if self._interleaver_type == InterleaverType.REGULAR:
            return "Regular"
        elif self._interleaver_type == InterleaverType.RANDOM:
            return "Random"
        elif self._interleaver_type == InterleaverType.SRANDOM:
            return "S-Random"

        return "None"

    def set_s_param(self, i_s_param):
        self._d_s_param = i_s_param

    def get_s_param(self):
        return self._d_s_param
