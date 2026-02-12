from matrix_sparse import SparseMatrix

try:
    import numpy as np
    from scipy import sparse
except ImportError as e:
    raise ImportError(
        "numpy and scipy are required but not installed. "
        "Please install them using: pip install numpy scipy"
    ) from e


def parse_string_to_int_array(s):
    """Parse a string of space-separated integers into a list."""
    if not s or not s.strip():
        return []
    # Split and filter out empty strings, then convert to int
    return [int(x) for x in s.split() if x.strip()]


def read_parity_check_matrix(file_name):
    """Read parity check matrix from file."""
    try:
        with open(file_name, 'r') as file:
            # Getting rows and cols count
            line = file.readline().strip()
            if not line:
                raise ValueError("Empty file or missing dimensions")
            arr_sizes = parse_string_to_int_array(line)
            
            if len(arr_sizes) < 2:
                raise ValueError("Invalid format: missing dimensions")

            # arr_sizes[0] - cols count (N)
            i_cols_cnt = arr_sizes[0]
            # arr_sizes[1] - rows count (M)
            i_rows_cnt = arr_sizes[1]

            if i_cols_cnt <= 0 or i_rows_cnt <= 0:
                raise ValueError(f"Invalid dimensions: cols={i_cols_cnt}, rows={i_rows_cnt}")

            # Skip next line (max weight: max column weight, max row weight)
            max_weights_line = file.readline()
            
            # array 1's of each col (column weights) - should have N values
            cols_weights_line = file.readline()
            if not cols_weights_line:
                raise ValueError("Unexpected end of file: missing column weights")
            arr_cols1s = parse_string_to_int_array(cols_weights_line)
            if len(arr_cols1s) != i_cols_cnt:
                raise ValueError(
                    f"Column weights count mismatch: expected {i_cols_cnt}, got {len(arr_cols1s)}. "
                    f"Line content: '{cols_weights_line.strip()}'"
                )

            # array 1's of each row (row weights) - should have M values
            rows_weights_line = file.readline()
            if not rows_weights_line:
                raise ValueError("Unexpected end of file: missing row weights")
            arr_rows1s = parse_string_to_int_array(rows_weights_line)
            if len(arr_rows1s) != i_rows_cnt:
                raise ValueError(
                    f"Row weights count mismatch: expected {i_rows_cnt}, got {len(arr_rows1s)}. "
                    f"Line content: '{rows_weights_line.strip()}'"
                )

            # Read column indices (skip them - we don't need them for row-based format)
            i_col = 0
            while i_col < i_cols_cnt:
                line = file.readline()
                if not line:
                    raise ValueError(f"Unexpected end of file while reading column {i_col}")
                i_col += 1

            # Read row indices and build sparse matrix directly (COO format)
            row_indices = []
            col_indices = []
            
            i_row = 0
            while i_row < len(arr_rows1s):
                line = file.readline()
                if not line:
                    raise ValueError(f"Unexpected end of file while reading row {i_row}")
                line = line.strip()
                if not line:
                    i_row += 1
                    continue
                    
                arr_row1s_indexes = parse_string_to_int_array(line)

                for idx in arr_row1s_indexes:
                    if idx == 0:
                        continue
                    if idx < 1 or idx > i_cols_cnt:
                        raise ValueError(f"Invalid column index {idx} in row {i_row} (valid range: 1-{i_cols_cnt})")

                    row_indices.append(i_row)
                    col_indices.append(idx - 1)  # Convert from 1-based to 0-based

                i_row += 1

            # Create sparse matrix directly from coordinates (much faster)
            data = np.ones(len(row_indices), dtype=np.int32)
            H_sparse = sparse.coo_matrix((data, (row_indices, col_indices)), 
                                        shape=(i_rows_cnt, i_cols_cnt), dtype=np.int32)
            H = SparseMatrix(sparse_matrix=H_sparse.tocsr())

        return H
    except Exception as e:
        print(f"Error: Could not read parity check matrix from file {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return SparseMatrix()
