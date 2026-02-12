from matrix_sparse import SparseMatrix
from utils import read_parity_check_matrix


def mod_inverse(a, q):
    """Function for modular inverse, if working over fields larger than q=2."""
    for i in range(1, q):
        if (a * i) % q == 1:
            return i
    return 1  # if q == 2, or if no inverse exists, return 1


def gaussian_elimination(matrix, columns=None, diagonalize=True, successful_cols=None, q=2):
    """Gaussian Elimination function in GF(q) arithmetic.
    Optimized for sparse matrices by using LIL format once for all operations.
    """
    import numpy as np
    from scipy import sparse
    
    nrows = matrix.get_rows()
    ncols = matrix.get_cols()
    cur_row = 0
    num_successful_cols = 0

    if successful_cols is None:
        successful_cols = [-1] * nrows

    if columns is None:
        columns = list(range(ncols))

    # Convert to LIL format once for efficient element-wise operations
    # LIL format is optimized for modifying individual elements
    lil_matrix = matrix.get_sparse_matrix().tolil()

    col_index = 0

    while True:
        if col_index >= len(columns):
            break

        cur_col = columns[col_index]
        pivot_row = -1

        # Search for a pivot row
        # Access row directly from LIL format (more efficient)
        for row in range(cur_row, nrows):
            if lil_matrix[row, cur_col] != 0:
                pivot_row = row
                break

        if pivot_row == -1:
            # This column is linearly dependent; continue with the next column
            col_index += 1
            continue

        # Swap rows if needed (swap entire rows in LIL format)
        if pivot_row > cur_row:
            # Swap rows in LIL format
            temp_row = lil_matrix.data[pivot_row].copy(), lil_matrix.rows[pivot_row].copy()
            lil_matrix.data[pivot_row] = lil_matrix.data[cur_row]
            lil_matrix.rows[pivot_row] = lil_matrix.rows[cur_row]
            lil_matrix.data[cur_row] = temp_row[0]
            lil_matrix.rows[cur_row] = temp_row[1]

        # Pivoting: set the pivot element to 1, if needed
        pivot_val = lil_matrix[cur_row, cur_col]
        if pivot_val > 1:
            if q > 2:
                factor = mod_inverse(pivot_val, q)
                # Multiply entire row by factor
                row_data = lil_matrix.data[cur_row]
                row_cols = lil_matrix.rows[cur_row]
                for idx, col in enumerate(row_cols):
                    row_data[idx] = (row_data[idx] * factor) % q

        # Eliminate the current column for all rows below the pivot
        pivot_row_data = lil_matrix.data[cur_row]
        pivot_row_cols = lil_matrix.rows[cur_row]
        pivot_row_dict = dict(zip(pivot_row_cols, pivot_row_data))
        
        for row in range(cur_row + 1, nrows):
            val = lil_matrix[row, cur_col]
            if val != 0:
                row_data = lil_matrix.data[row]
                row_cols = lil_matrix.rows[row]
                row_dict = dict(zip(row_cols, row_data))
                
                # Combine rows
                if q == 2:
                    # XOR operation in GF(2)
                    for col, pivot_val in pivot_row_dict.items():
                        if col in row_dict:
                            new_val = (row_dict[col] ^ pivot_val) % 2
                        else:
                            new_val = pivot_val % 2
                        if new_val != 0:
                            row_dict[col] = new_val
                        elif col in row_dict:
                            del row_dict[col]
                else:
                    # General GF(q) operation
                    for col, pivot_val in pivot_row_dict.items():
                        if col in row_dict:
                            new_val = (row_dict[col] - val * pivot_val + q) % q
                        else:
                            new_val = (-val * pivot_val + q) % q
                        if new_val != 0:
                            row_dict[col] = new_val
                        elif col in row_dict:
                            del row_dict[col]
                
                # Update LIL row
                if row_dict:
                    cols = sorted(row_dict.keys())
                    lil_matrix.rows[row] = cols
                    lil_matrix.data[row] = [row_dict[col] for col in cols]
                else:
                    lil_matrix.rows[row] = []
                    lil_matrix.data[row] = []

        successful_cols[num_successful_cols] = cur_col
        num_successful_cols += 1

        if num_successful_cols == nrows:
            break

        cur_row += 1
        col_index += 1

    if diagonalize:
        # Make the matrix diagonal (zero above pivots)
        for diag_col in range(num_successful_cols):
            cur_col = successful_cols[diag_col]
            diag_row_data = lil_matrix.data[diag_col]
            diag_row_cols = lil_matrix.rows[diag_col]
            diag_row_dict = dict(zip(diag_row_cols, diag_row_data))
            
            for row in range(diag_col):
                val = lil_matrix[row, cur_col]
                if val != 0:
                    row_data = lil_matrix.data[row]
                    row_cols = lil_matrix.rows[row]
                    row_dict = dict(zip(row_cols, row_data))
                    
                    # Combine rows
                    if q == 2:
                        # XOR operation in GF(2)
                        for col, diag_val in diag_row_dict.items():
                            if col in row_dict:
                                new_val = (row_dict[col] ^ diag_val) % 2
                            else:
                                new_val = diag_val % 2
                            if new_val != 0:
                                row_dict[col] = new_val
                            elif col in row_dict:
                                del row_dict[col]
                    else:
                        # General GF(q) operation
                        for col, diag_val in diag_row_dict.items():
                            if col in row_dict:
                                new_val = (row_dict[col] - val * diag_val + q) % q
                            else:
                                new_val = (-val * diag_val + q) % q
                            if new_val != 0:
                                row_dict[col] = new_val
                            elif col in row_dict:
                                del row_dict[col]
                    
                    # Update LIL row
                    if row_dict:
                        cols = sorted(row_dict.keys())
                        lil_matrix.rows[row] = cols
                        lil_matrix.data[row] = [row_dict[col] for col in cols]
                    else:
                        lil_matrix.rows[row] = []
                        lil_matrix.data[row] = []

    # Convert back to CSR and update matrix
    matrix.update_from_sparse_matrix(lil_matrix.tocsr())

    # Resize successful_cols to contain only the necessary values
    successful_cols = successful_cols[:num_successful_cols]
    return successful_cols


class EncoderDecoderData:
    def __init__(self, file_parity_check_matrix_name):
        self._h = read_parity_check_matrix(file_parity_check_matrix_name)

        self._n = self._h.get_cols()
        self._m = self._h.get_rows()
        self._k = self._n - self._m

        if self._n == 0:
            raise ValueError("Invalid parity check matrix: matrix is empty")
        
        self._rate = float(self._k) / self._n if self._n > 0 else 0.0

        permutation_vec = []
        self._h_std = self.create_standart_parity_check_matrix(self._h, permutation_vec)
        self._permutation = permutation_vec

        self._g = self.create_generator_matrix(self._h_std)
        
        # Pre-compute and cache G^T for faster encoding
        # G is k x n, so G^T is n x k
        self._g_transpose = self._g.transpose()
        
        # Richardson-Urbanke encoding data (initialized on demand)
        self._ru_data = None
        
        # Pre-compute and cache decoder structures for multi-processing efficiency
        # These structures are read-only and can be shared across processes via pickle
        # Note: scipy.sparse matrices are efficiently pickle-able for multiprocessing
        self._h_sparse_cached = self._h_std.get_sparse_matrix()  # Cache sparse matrix reference
        self._h_coo = None
        self._var_to_check = None
        self._check_to_var = None
        self._decoder_structures_initialized = False

        # Verify generator matrix (G * H^T should be zero matrix mod 2)
        # This is equivalent to checking that G * H^T = 0 (all zeros)
        mtx = self._g.multiply(self._h_std.transpose())
        mtx_sparse = mtx.get_sparse_matrix()
        
        # After mod 2 operation in multiply(), check if result is zero matrix
        # In C++ code, sum of all elements is checked (should be 0)
        # For sparse matrices, we check both nnz and sum
        try:
            # Calculate sum of all elements (as in C++ code)
            # For sparse matrix, sum() gives sum of all non-zero elements
            # But we need sum of ALL elements (including zeros)
            # So we convert to dense for accurate sum calculation
            dense_sum = float(mtx_sparse.sum())
            
            # In GF(2), sum mod 2 should be 0 for valid generator matrix
            # Also check number of non-zero elements
            if mtx_sparse.nnz > 0 or (dense_sum % 2) != 0:
                # Matrix is not zero - invalid generator matrix
                coo = mtx_sparse.tocoo()
                non_zero_count = mtx_sparse.nnz
                
                # Get some sample non-zero elements for debugging
                sample_elements = []
                if coo.data.size > 0:
                    for i in range(min(5, len(coo.data))):
                        sample_elements.append(f"({coo.row[i]},{coo.col[i]})={coo.data[i]}")
                
                error_msg = (
                    f"Invalid generator matrix: G * H^T != 0. "
                    f"Sum of all elements = {dense_sum} (expected 0 mod 2). "
                    f"Non-zero elements: {non_zero_count}. "
                    f"Matrix dimensions: {mtx_sparse.shape[0]}x{mtx_sparse.shape[1]}"
                )
                if sample_elements:
                    error_msg += f". Sample non-zero elements: {', '.join(sample_elements)}"
                
                raise ValueError(error_msg)
        except ValueError:
            raise
        except Exception as e:
            # If sum calculation fails, check nnz as fallback
            if mtx_sparse.nnz > 0:
                raise ValueError(
                    f"Invalid generator matrix: G * H^T != 0. "
                    f"Found {mtx_sparse.nnz} non-zero elements."
                ) from e

    def create_standart_parity_check_matrix(self, H, permutation_vec):
        """Method Gauss-Jordan to get H in the form [A | I_m]."""
        # Copy sparse matrix
        H_std = SparseMatrix(sparse_matrix=H.get_sparse_matrix().copy())

        successful_cols = gaussian_elimination(H_std)

        # Check if matrix has full rank
        rank = len(successful_cols)
        expected_rank = H_std.get_rows()
        
        if expected_rank != rank:
            # The matrix doesn't have full rank
            # This means some rows are linearly dependent
            # After Gaussian elimination with diagonalize=True, the first 'rank' rows should be independent
            print(f"Warning: Matrix rank is {rank}, expected {expected_rank}. Some rows are linearly dependent.")
            print("Using only independent rows to create standard form...")
            
            # After Gaussian elimination, the first 'rank' rows should be the independent ones
            # Create new matrix with only the first 'rank' rows
            H_std = H_std.extract_sub_matrix(0, 0, rank, H_std.get_cols())
            
            # Re-run Gaussian elimination to ensure proper form
            successful_cols = gaussian_elimination(H_std)
            
            if len(successful_cols) != rank:
                raise ValueError(
                    f"Internal error: After removing dependent rows, rank is {len(successful_cols)}, "
                    f"expected {rank}. This should not happen."
                )
            
            # Update parameters
            self._m = rank
            self._k = self._n - self._m
            self._rate = float(self._k) / self._n if self._n > 0 else 0.0
            
            print(f"Adjusted parameters: n={self._n}, m={self._m}, k={self._k}, rate={self._rate:.4f}")
            
        for i_col in range(H_std.get_cols()):
            if i_col in successful_cols:
                continue

            permutation_vec.append(i_col)

        permutation_vec.extend(successful_cols)

        H_std.permute_columns(permutation_vec)

        return H_std

    def create_generator_matrix(self, H):
        """Create generator matrix from parity check matrix.
        H should be in standard form [A | I_m] where:
        - A is m x k matrix (first m rows, first k columns)
        - I_m is m x m identity matrix (last m columns)
        G = [I_k | A^T] where I_k is k x k identity matrix
        """
        # Verify H is in correct form
        if H.get_rows() != self._m or H.get_cols() != self._n:
            raise ValueError(
                f"Matrix H dimensions mismatch: expected {self._m}x{self._n}, "
                f"got {H.get_rows()}x{H.get_cols()}"
            )
        
        # Extract A: first m rows, first k columns
        # H should be [A | I_m], so A is columns 0 to k-1
        A = H.extract_sub_matrix(0, 0, self._m, self._k)

        # Create identity matrix I_k (k x k)
        I = SparseMatrix.create_identity_matrix(self._k)

        # G = [I_k | A^T]
        # A^T is k x m, so concatenation gives k x (k + m) = k x n
        G = I.concatenate_horizontally(A.transpose())

        return G
    
    def _convert_to_lower_triangular_form(self, H, max_gap=None, target_gap=None):
        """Convert H to approximate lower triangular form with minimal gap.
        
        Richardson-Urbanke preprocessing algorithm:
        1. Find row and column permutations that minimize gap
        2. Gap g is the number of rows that cannot be placed in triangular T
        3. Goal: minimize g to get H in form [A B T; C D E] where T is lower triangular
        
        Algorithm (greedy approach):
        - Try to build lower triangular T from bottom-right corner
        - Work backwards: for each diagonal position, find a row/column pair
        - If successful, gap = 0; otherwise increase gap
        
        Args:
            H: Parity check matrix to convert
            max_gap: Maximum gap to try (default: min(m//2, 20))
            target_gap: Target gap to use (if specified, tries only this gap)
        
        Returns:
            tuple: (H_ru, row_permutation, col_permutation, gap)
            where H_ru is the permuted matrix in approximate lower triangular form
        """
        import numpy as np
        from scipy import sparse
        from matrix_sparse import SparseMatrix
        
        m = H.get_rows()
        n = H.get_cols()
        
        # Get sparse matrix in COO format for efficient access
        H_sparse = H.get_sparse_matrix()
        H_coo = H_sparse.tocoo()
        
        # Build row and column index lists for efficient lookup
        # For each row, get list of columns with 1s
        row_to_cols = {}
        for i, j in zip(H_coo.row, H_coo.col):
            if i not in row_to_cols:
                row_to_cols[i] = []
            row_to_cols[i].append(j)
        
        # For each column, get list of rows with 1s
        col_to_rows = {}
        for i, j in zip(H_coo.row, H_coo.col):
            if j not in col_to_rows:
                col_to_rows[j] = []
            col_to_rows[j].append(i)
        
        # Greedy algorithm: try to minimize gap
        # Start with gap = 0 and increase if needed
        if max_gap is None:
            max_gap = min(m // 2, 20)  # Reasonable limit
        
        best_gap = m
        best_row_perm = list(range(m))
        best_col_perm = list(range(n))
        
        # If target_gap is specified, try only that gap
        if target_gap is not None:
            # Validate target_gap
            if 0 <= target_gap <= max_gap:
                gap_range = [target_gap]
            else:
                # If target_gap is out of range, use max_gap
                gap_range = [max_gap]
        else:
            # Try different gap values starting from 0
            gap_range = range(max_gap + 1)
        
        # Try different gap values
        for current_gap in gap_range:
            t_size = m - current_gap  # Size of T matrix
            
            # Try to build T matrix (lower triangular) of size t_size x t_size
            # T should be in the bottom-right corner of H_ru
            # Strategy: work from bottom-right diagonal upwards
            
            used_rows = set()
            used_cols = set()
            t_row_mapping = {}  # Maps T row index -> H row index
            t_col_mapping = {}  # Maps T col index -> H col index
            
            # Try to fill T matrix from bottom-right to top-left
            success = True
            for diag_pos in range(t_size):
                t_row = t_size - 1 - diag_pos  # T row index (from bottom)
                t_col = t_size - 1 - diag_pos  # T col index (from right)
                
                # Find a row in H that:
                # 1. Has a 1 in a column that can be placed at t_col
                # 2. Can form lower triangular structure (1s only at/below diagonal)
                found = False
                
                # Try to find a suitable row
                for h_row in range(m):
                    if h_row in used_rows:
                        continue
                    
                    # Get columns with 1s in this row
                    if h_row not in row_to_cols:
                        continue
                    
                    row_cols = row_to_cols[h_row]
                    
                    # Check if this row can be placed at t_row in T
                    # For lower triangular T: row t_row should have 1s only at positions <= t_col
                    # We use a greedy heuristic: if this row has an unused column, try to place it
                    # This is not optimal but should work for many cases
                    for h_col in row_cols:
                        if h_col in used_cols:
                            continue
                        
                        # Count how many columns in this row are already used
                        unused_cols_in_row = [c for c in row_cols if c not in used_cols]
                        
                        # If we have at least one unused column (h_col), we can try to place it
                        if len(unused_cols_in_row) > 0:
                            # Found a match - use greedy approach
                            # We'll place h_col at t_col and hope other columns can be arranged
                            t_row_mapping[t_row] = h_row
                            t_col_mapping[t_col] = h_col
                            used_rows.add(h_row)
                            used_cols.add(h_col)
                            found = True
                            break
                    
                    if found:
                        break
                
                if not found:
                    # Cannot build T with this gap
                    success = False
                    break
            
            if success and len(t_row_mapping) == t_size:
                # Successfully built T with gap = current_gap
                best_gap = current_gap
                
                # Build full permutations
                # For rows: first (m-g) rows go to T positions, last g rows stay
                row_perm = [0] * m
                for t_row, h_row in t_row_mapping.items():
                    row_perm[t_row] = h_row
                
                # Fill remaining rows (gap rows)
                remaining_rows = [r for r in range(m) if r not in used_rows]
                for idx, h_row in enumerate(remaining_rows):
                    row_perm[t_size + idx] = h_row
                
                # For columns: info columns first, then gap columns, then T columns
                # Standard form already has good column ordering
                # Use standard form column permutation as base
                col_perm = self._permutation.copy()
                
                # Store best solution
                best_row_perm = row_perm
                best_col_perm = col_perm
                break
        
        # Apply permutations to create H_ru
        # If we found a solution with gap < m, use it
        # Otherwise, fall back to standard form (gap = 0)
        if best_gap < m:
            # Create a copy of H for permutation
            from matrix_sparse import SparseMatrix
            H_copy = SparseMatrix(sparse_matrix=self._h.get_sparse_matrix().copy())
            
            # Apply row permutation
            H_copy.permute_rows(best_row_perm)
            
            # Apply column permutation
            H_copy.permute_columns(best_col_perm)
            
            H_ru = H_copy
            row_permutation = best_row_perm
            col_permutation = best_col_perm
            gap = best_gap
        else:
            # Fall back to standard form (gap = 0)
            # For gap=0, H_std should be in form [A | I_m] where:
            # - First k columns are info columns (A)
            # - Last m columns are parity columns (I_m)
            # 
            # However, self._permutation may not guarantee this structure.
            # We need to ensure that first k positions contain info columns.
            # 
            # Check if self._permutation has the correct structure
            k = self._k
            info_cols_in_first_k = sum(1 for i in range(k) if self._permutation[i] < k)
            
            if info_cols_in_first_k < k:
                # The permutation doesn't guarantee that first k positions are info columns
                # This is a structural problem: H_std is not in correct [A | I_m] form
                # 
                # For gap=0, we should use H_std as-is, but we need to warn that
                # the structure is incorrect and encoding may fail
                col_permutation = self._permutation.copy()
                
                print(f"WARNING: For gap=0, only {info_cols_in_first_k} info columns in first k={k} positions.")
                print(f"  H_std may not be in correct [A | I_m] form. Richardson-Urbanke encoding may fail.")
                print(f"  Consider using standard encoding method instead.")
            else:
                col_permutation = self._permutation.copy()
            
            H_ru = self._h_std
            row_permutation = list(range(m))
            gap = 0
        
        return (H_ru, row_permutation, col_permutation, gap)
    
    def prepare_richardson_urbanke_encoding(self, gap=None):
        """Prepare matrix H in Richardson-Urbanke form.
        
        Richardson-Urbanke encoding requires H to be in approximate lower triangular form:
        H = [A B T; C D E] where T is lower triangular with ones on diagonal.
        
        This function attempts to minimize gap g by finding optimal row/column permutations,
        or uses the specified gap if provided.
        
        Args:
            gap: Optional gap value to use. If None, finds minimal gap automatically.
        
        Returns a dictionary with submatrices and permutation for encoding.
        """
        import numpy as np
        from scipy import sparse
        from matrix_sparse import SparseMatrix
        
        # If gap is specified and ru_data already exists with same gap, return it
        if self._ru_data is not None:
            if gap is None or self._ru_data.get('gap') == gap:
                return self._ru_data
            # If gap changed, need to recompute
            self._ru_data = None
        
        m = self._m
        n = self._n
        k = self._k
        
        # Try to convert to lower triangular form with specified or minimal gap
        if gap is not None:
            # Use specified gap
            H_ru, row_perm, col_perm, found_gap = self._convert_to_lower_triangular_form(self._h, target_gap=gap)
            if found_gap != gap:
                print(f"Предупреждение: Не удалось использовать заданный gap={gap}, используется gap={found_gap}")
                gap = found_gap
        else:
            # Find minimal gap automatically
            H_ru, row_perm, col_perm, gap = self._convert_to_lower_triangular_form(self._h)
        
        # Extract submatrices based on gap
        # H_ru structure: [A B T; C D E]
        # where:
        # - A: (m-g) x k
        # - B: (m-g) x g  
        # - T: (m-g) x (m-g) lower triangular
        # - C: g x k
        # - D: g x g
        # - E: g x (m-g)
        
        t_size = m - gap  # Size of T matrix
        
        # Extract submatrices
        # A: first (m-g) rows, first k columns
        A = H_ru.extract_sub_matrix(0, 0, t_size, k)
        
        # B: first (m-g) rows, columns k to k+g-1
        if gap > 0:
            B = H_ru.extract_sub_matrix(0, k, t_size, gap)
        else:
            from matrix_sparse import SparseMatrix
            B = SparseMatrix(t_size, 0)
        
        # T: first (m-g) rows, columns k+g to k+g+(m-g)-1 = n-g to n-1
        T = H_ru.extract_sub_matrix(0, k + gap, t_size, t_size)
        
        # C: last g rows, first k columns
        if gap > 0:
            C = H_ru.extract_sub_matrix(t_size, 0, gap, k)
        else:
            from matrix_sparse import SparseMatrix
            C = SparseMatrix(0, k)
        
        # D: last g rows, columns k to k+g-1
        if gap > 0:
            D = H_ru.extract_sub_matrix(t_size, k, gap, gap)
        else:
            from matrix_sparse import SparseMatrix
            D = SparseMatrix(0, 0)
        
        # E: last g rows, columns k+g to n-1
        if gap > 0:
            E = H_ru.extract_sub_matrix(t_size, k + gap, gap, t_size)
        else:
            from matrix_sparse import SparseMatrix
            E = SparseMatrix(0, t_size)
        
        # Compute φ = D + E * T^(-1) * B (mod 2)
        # For gap = 0: phi is not needed
        if gap > 0:
            # Compute T^(-1) * B using forward substitution for each column of B
            # T is lower triangular, so we can solve T * X = B column by column
            # Then compute E * (T^(-1) * B)
            # Finally: phi = D + E * T^(-1) * B
            
            # Get sparse matrices
            T_sparse = T.get_sparse_matrix()
            T_dense = (T_sparse.toarray() % 2).astype(np.int32)
            B_sparse = B.get_sparse_matrix()
            E_sparse = E.get_sparse_matrix()
            D_sparse = D.get_sparse_matrix()
            
            # Compute T^(-1) * B column by column using forward substitution
            # For each column b of B, solve T * x = b
            B_dense = (B_sparse.toarray() % 2).astype(np.int32)
            T_inv_B = np.zeros((t_size, gap), dtype=np.int32)
            
            for col_idx in range(gap):
                b_col = B_dense[:, col_idx]
                x_col = np.zeros(t_size, dtype=np.int32)
                
                # Forward substitution: T * x = b (mod 2)
                for i in range(t_size):
                    sum_val = 0
                    for j in range(i):
                        sum_val = (sum_val + T_dense[i, j] * x_col[j]) % 2
                    x_col[i] = (b_col[i] + sum_val) % 2
                
                T_inv_B[:, col_idx] = x_col
            
            # Compute E * T^(-1) * B (mod 2)
            # E_sparse is sparse, T_inv_B is dense numpy array
            E_T_inv_B_sparse = E_sparse.dot(sparse.csr_matrix(T_inv_B))
            # Convert to dense array and apply mod 2
            if hasattr(E_T_inv_B_sparse, 'toarray'):
                E_T_inv_B = (E_T_inv_B_sparse.toarray() % 2).astype(np.int32)
            else:
                E_T_inv_B = (E_T_inv_B_sparse % 2).astype(np.int32)
            
            # Compute phi = D + E * T^(-1) * B (mod 2)
            D_dense = (D_sparse.toarray() % 2).astype(np.int32)
            phi_dense = (D_dense + E_T_inv_B) % 2
            
            # Convert back to SparseMatrix
            phi = SparseMatrix(sparse_matrix=sparse.csr_matrix(phi_dense))
        else:
            phi = None
        
        # Store original H matrix (before permutation) for verification
        H_original_sparse = self._h.get_sparse_matrix()
        
        # Store RU encoding data
        self._ru_data = {
            'permutation': col_perm.copy(),  # Column permutation
            'row_permutation': row_perm.copy(),  # Row permutation (for reference)
            'A': A,
            'B': B,
            'T': T,
            'C': C,
            'D': D,
            'E': E,
            'phi': phi,
            'H_ru': H_ru,  # H in RU form (after permutations)
            'H_original_sparse': H_original_sparse,  # Original H (for verification)
            'gap': gap,
            'm': m,
            'n': n,
            'k': k
        }
        
        return self._ru_data
    
    def _init_decoder_structures(self):
        """Initialize decoder structures (COO format and neighbor structures).
        This is called once and cached for multi-threading efficiency.
        """
        if self._decoder_structures_initialized:
            return
        
        # Use cached sparse matrix H for efficient access
        h_sparse = self._h_sparse_cached
        
        # Convert to COO format (once, cached)
        self._h_coo = h_sparse.tocoo()
        
        # Precompute neighbor structures for efficient iteration
        # Variable nodes -> check nodes
        self._var_to_check = {}
        # Check nodes -> variable nodes
        self._check_to_var = {}
        
        # Build neighbor structures from COO format
        for i, j in zip(self._h_coo.row, self._h_coo.col):
            if j not in self._var_to_check:
                self._var_to_check[j] = []
            self._var_to_check[j].append(i)
            
            if i not in self._check_to_var:
                self._check_to_var[i] = []
            self._check_to_var[i].append(j)
        
        self._decoder_structures_initialized = True
    
    def get_decoder_structures(self):
        """Get cached decoder structures for multi-threading.
        Returns tuple: (H_coo, var_to_check, check_to_var)
        """
        if not self._decoder_structures_initialized:
            self._init_decoder_structures()
        return (self._h_coo, self._var_to_check, self._check_to_var)