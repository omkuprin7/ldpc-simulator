class Matrix:
    def __init__(self, rows=0, cols=0, default_value=0, values=None):
        if values is not None:
            self._data = values
            self._rows = len(values)
            self._cols = len(values[0]) if values else 0
        else:
            self._rows = rows
            self._cols = cols
            self._data = [[default_value for _ in range(cols)] for _ in range(rows)]

    def get_rows(self):
        """Get the number of rows (M). M = N - K. Parity bits."""
        return self._rows

    def get_cols(self):
        """Get the number of columns (N)."""
        return self._cols

    def get_message_bit_length(self):
        """Get the message length (K). K = N - M"""
        return self._cols - self._rows

    def add_row(self, row):
        """Add a row to the matrix."""
        if self._cols != len(row):
            raise ValueError("Must have the same dimensions for addition.")
        self._data.append(row)
        self._rows += 1

    def add(self, mtx):
        """Add two matrices."""
        if self._rows != mtx.get_rows() or self._cols != mtx.get_cols():
            raise ValueError("Matrices must have the same dimensions for addition.")

        result_mtx = Matrix(self._rows, self._cols)
        for i_row in range(self._rows):
            for i_col in range(self._cols):
                result_mtx._data[i_row][i_col] = self._data[i_row][i_col] + mtx._data[i_row][i_col]

        return result_mtx

    def multiply(self, mtx):
        """Multiply two matrices (mod 2)."""
        if self._cols != mtx.get_rows():
            raise ValueError(
                "Number of columns of the first matrix must be equal to the number of rows of the second matrix."
            )

        result_mtx = Matrix(self._rows, mtx.get_cols())
        for i_row in range(self._rows):
            for i_col_mtx in range(mtx.get_cols()):
                d_res = 0
                for i_col in range(self._cols):
                    d_res += self._data[i_row][i_col] * mtx._data[i_col][i_col_mtx]

                result_mtx._data[i_row][i_col_mtx] = d_res % 2

        return result_mtx

    def print(self):
        """Print the matrix."""
        for row in self._data:
            print(" ".join(str(value) for value in row))

    def set_element(self, i_row, i_col, i_value):
        """Set an element at a specific position (iRow, iCol)."""
        if i_row >= self._rows or i_col >= self._cols or i_row < 0 or i_col < 0:
            raise IndexError("Index out of bounds")

        self._data[i_row][i_col] = i_value

    def get_element(self, i_row, i_col):
        """Get an element at a specific position (iRow, iCol)."""
        if i_row >= self._rows or i_col >= self._cols or i_row < 0 or i_col < 0:
            raise IndexError("Index out of bounds")

        return self._data[i_row][i_col]

    def transpose(self):
        """Transpose the matrix."""
        transposed_mtx = Matrix(self._cols, self._rows)
        for i_row in range(self._rows):
            for i_col in range(self._cols):
                transposed_mtx.set_element(i_col, i_row, self._data[i_row][i_col])

        return transposed_mtx

    def permute_columns(self, permutation):
        """Permute columns of the matrix."""
        if len(permutation) != self._cols:
            raise ValueError("Invalid permutation size")

        permuted_matrix = [[0 for _ in range(self._cols)] for _ in range(self._rows)]

        for i_row in range(self._rows):
            for i_col in range(self._cols):
                permuted_matrix[i_row][i_col] = self._data[i_row][permutation[i_col]]

        self._data = permuted_matrix

    def swap_rows(self, row1, row2):
        """Swap two rows in the matrix."""
        if row1 >= self._rows or row2 >= self._rows or row1 < 0 or row2 < 0:
            raise IndexError("Row index out of bounds")

        self._data[row1], self._data[row2] = self._data[row2], self._data[row1]

    def extract_sub_matrix(self, start_row, start_col, sub_rows, sub_cols):
        """Extract a sub-matrix."""
        if start_row < 0 or start_col < 0 or start_row + sub_rows > self._rows or start_col + sub_cols > self._cols:
            raise IndexError("Sub-matrix dimensions are out of bounds")

        sub_matrix = Matrix(sub_rows, sub_cols)

        for i in range(sub_rows):
            for j in range(sub_cols):
                sub_matrix.set_element(i, j, self._data[start_row + i][start_col + j])

        return sub_matrix

    @staticmethod
    def create_identity_matrix(size):
        """Create an identity matrix."""
        identity = Matrix(size, size)

        for i_index in range(size):
            identity.set_element(i_index, i_index, 1)

        return identity

    def concatenate_horizontally(self, mtx):
        """Concatenate matrices horizontally."""
        if self._rows != mtx._rows:
            raise ValueError("Row counts must match for horizontal concatenation")

        result_mtx = Matrix(self._rows, self._cols + mtx._cols)

        for i_row in range(self._rows):
            for i_col in range(self._cols):
                result_mtx.set_element(i_row, i_col, self._data[i_row][i_col])

            for i_col in range(mtx._cols):
                result_mtx.set_element(i_row, self._cols + i_col, mtx._data[i_row][i_col])

        return result_mtx

    def concatenate_vertically(self, mtx):
        """Concatenate matrices vertically."""
        if self._cols != mtx._cols:
            raise ValueError("Column counts must match for vertical concatenation")

        result_mtx = Matrix(self._rows + mtx._rows, self._cols)

        for i_row in range(self._rows):
            for i_col in range(self._cols):
                result_mtx.set_element(i_row, i_col, self._data[i_row][i_col])

        for i_row in range(mtx._rows):
            for i_col in range(mtx._cols):
                result_mtx.set_element(self._rows + i_row, i_col, mtx._data[i_row][i_col])

        return result_mtx

    def init(self, rows, cols, default_value=0):
        """Initialize the matrix."""
        self._rows = rows
        self._cols = cols
        self._data = [[default_value for _ in range(cols)] for _ in range(rows)]

    def get_data(self):
        """Get the matrix data."""
        return self._data
