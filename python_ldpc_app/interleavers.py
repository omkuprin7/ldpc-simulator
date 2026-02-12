import math
import random
from enums import InterleaverType


class Interleaver:
    def __init__(self, interleaver_type=InterleaverType.NONE):
        self.m_e_interleaver_type = interleaver_type

    def set_interleaver_type(self, interleaver_type):
        self.m_e_interleaver_type = interleaver_type

    def get_interleaver_type(self):
        return self.m_e_interleaver_type

    def interleave(self, databuffer):
        """Interleave data in the buffer."""
        pass

    def deinterleave(self, databuffer):
        """Deinterleave data in the buffer."""
        pass


class RandomInterleaver(Interleaver):
    def __init__(self):
        super().__init__(InterleaverType.RANDOM)

    def interleave(self, databuffer):
        """Random interleaving."""
        arr_indexes = list(range(len(databuffer._encoded_data)))
        databuffer._interleaving_pos_indexes = []

        while arr_indexes:
            i_position = random.randint(0, len(arr_indexes) - 1)

            databuffer._interleaving_pos_indexes.append(arr_indexes[i_position])

            arr_indexes.pop(i_position)

        interleaving_data = [0] * len(databuffer._encoded_data)
        for i_index in range(len(databuffer._encoded_data)):
            interleaving_data[i_index] = databuffer._encoded_data[databuffer._interleaving_pos_indexes[i_index]]

        databuffer._encoded_data = interleaving_data

    def deinterleave(self, databuffer):
        """Random deinterleaving."""
        deinterleaver_channel_data = [0.0] * len(databuffer._interleaving_pos_indexes)
        for i_index in range(len(databuffer._interleaving_pos_indexes)):
            deinterleaver_channel_data[databuffer._interleaving_pos_indexes[i_index]] = databuffer._channel_data[i_index]

        databuffer._channel_data = deinterleaver_channel_data


class RegularInterleaver(Interleaver):
    def __init__(self):
        super().__init__(InterleaverType.REGULAR)

    def calculate_rows_and_cols_for_regular_interleaver(self, databuffer):
        """Calculate rows and columns for regular interleaver."""
        i_size_encoded_data = len(databuffer._encoded_data)
        i_rows = int(math.sqrt(i_size_encoded_data))
        i_cols = 0
        while True:
            i_cols = i_size_encoded_data // i_rows
            if i_rows * i_cols != i_size_encoded_data:
                i_rows -= 1
            else:
                break

            if i_rows == 0:
                break

        return (i_rows, i_cols)

    def regular_interleave(self, databuffer, i_rows, i_cols):
        """Regular interleaving."""
        interleaving_data = [0] * len(databuffer._encoded_data)
        databuffer._interleaving_pos_indexes = []
        for i_row in range(i_rows):
            for i_col in range(i_cols):
                i_interleaved_index = i_col * i_rows + i_row
                interleaving_data[i_interleaved_index] = databuffer._encoded_data[i_row * i_cols + i_col]

                databuffer._interleaving_pos_indexes.append(i_interleaved_index)

        databuffer._encoded_data = interleaving_data

    def interleave(self, databuffer):
        """Regular interleaving."""
        rows_and_cols = self.calculate_rows_and_cols_for_regular_interleaver(databuffer)
        i_rows = rows_and_cols[0]
        i_cols = rows_and_cols[1]
        if (i_rows == 0) or (i_cols == 0):
            return

        self.regular_interleave(databuffer, i_rows, i_cols)

    def deinterleave(self, databuffer):
        """Regular deinterleaving."""
        deinterleaver_channel_data = []
        for i_index in range(len(databuffer._interleaving_pos_indexes)):
            deinterleaver_channel_data.append(databuffer._channel_data[databuffer._interleaving_pos_indexes[i_index]])

        databuffer._channel_data = deinterleaver_channel_data


class SRandomInterleaver(Interleaver):
    def __init__(self, i_s_param=2):
        super().__init__(InterleaverType.SRANDOM)
        self.m_i_s_param = i_s_param

    def set_s_param(self, i_s_param):
        self.m_i_s_param = i_s_param

    def get_s_param(self):
        return self.m_i_s_param

    def interleave(self, databuffer):
        """S-Random interleaving."""
        s = self.m_i_s_param
        n = len(databuffer._encoded_data)
        tmp = [0] * n
        tmp_pos = [0] * n

        i = 1
        while i <= n:
            # уменьшаем счетчики
            for j in range(n):
                if tmp[j] > 0:
                    tmp[j] -= 1
            c = 0
            for j in range(n):
                if tmp[j] == 0:
                    c += 1
            if c == 0:
                continue
            # ищем свободную ячейку
            z = random.randint(0, n - 1)
            while tmp[z] != 0:
                z += 1
                if z == n:
                    z = 0
            # устанавливаем счетчики
            tmp[z] = -1
            left = z - s + 1
            right = z + s - 1
            if left < 0:
                left = 0
            if right > n - 1:
                right = n - 1
            for j in range(left, right + 1):
                if tmp[j] != -1:
                    tmp[j] = s

            tmp_pos[i - 1] = z
            i += 1

        databuffer._interleaving_pos_indexes = tmp_pos

        interleaving_data = [0] * len(databuffer._encoded_data)
        for i_index in range(len(databuffer._encoded_data)):
            interleaving_data[i_index] = databuffer._encoded_data[databuffer._interleaving_pos_indexes[i_index]]

        databuffer._encoded_data = interleaving_data

    def deinterleave(self, databuffer):
        """S-Random deinterleaving."""
        deinterleaver_channel_data = [0.0] * len(databuffer._interleaving_pos_indexes)
        for i_index in range(len(databuffer._interleaving_pos_indexes)):
            deinterleaver_channel_data[databuffer._interleaving_pos_indexes[i_index]] = databuffer._channel_data[i_index]

        databuffer._channel_data = deinterleaver_channel_data
