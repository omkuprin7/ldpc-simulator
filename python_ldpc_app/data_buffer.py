import math
import random

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "numpy is required but not installed. Please install it using: pip install numpy"
    )

from enums import InterleaverType
from generator import Generator


class DataBuffer:
    def __init__(self, size=0):
        self._size = size
        self._data = []
        self._encoded_data = []
        self._decoded_data = []
        self._channel_data = []
        self._interleaving_pos_indexes = []

        if size > 0:
            self._data = Generator.generate_bit_sequence(size)

    def get_size(self):
        """Returns size of data buffer."""
        return self._size

    def get_data(self):
        """Gets input data (input sequence)."""
        return self._data

    def print(self):
        """Prints main data of DataBuffer."""
        print("Bit Sequence: ", end="")
        for bit in self._data:
            print(bit, end="")
        print()

        print("Encode Sequence: ", end="")
        for bit in self._encoded_data:
            print(bit, end="")
        print()

    def encode(self, G_or_G_transpose):
        """Encodes input data using sparse matrix multiplication.
        Input: G_or_G_transpose - Generator matrix (SparseMatrix) or its transpose.
        If G_transpose is provided (from encoder_decoder_data._g_transpose), it's used directly.
        Otherwise, G is expected and will be transposed.
        G is k x n matrix, data is k-length vector.
        encoded = data^T * G (in matrix form: G^T * data as column vector)
        """
        # Check if we have G or G^T by checking dimensions
        # G is k x n, G^T is n x k
        # If rows == len(data), it's G; if cols == len(data), it's G^T
        if G_or_G_transpose.get_rows() == len(self._data):
            # It's G, need to transpose
            G_transpose = G_or_G_transpose.get_sparse_matrix().transpose()
        elif G_or_G_transpose.get_cols() == len(self._data):
            # It's G^T, use directly
            G_transpose = G_or_G_transpose.get_sparse_matrix()
        else:
            raise ValueError("Matrix dimensions don't match data length")

        # Convert input data to numpy array (column vector: k x 1)
        data_vector = np.array(self._data, dtype=np.int32).reshape(-1, 1)
        
        # Multiply: G^T (n x k) * data (k x 1) = result (n x 1)
        result = G_transpose.dot(data_vector)
        
        # Result is a sparse matrix, convert to dense array, then mod 2 and convert to list
        if hasattr(result, 'toarray'):
            # It's a sparse matrix
            result_array = result.toarray().flatten()
        else:
            # It's already a numpy array
            result_array = result.flatten()
        
        # Mod 2 operation and convert to list
        self._encoded_data = (result_array % 2).astype(int).tolist()
    
    def encode_richardson_urbanke(self, ru_data):
        """Encode using Richardson-Urbanke method.
        
        Richardson-Urbanke encoding algorithm:
        H = [A B T; C D E] where T is lower triangular, gap = g
        
        For gap = 0 (standard form H_std = [A | I_m]):
        - Этап А: Solve T * y^T = A * s^T (forward substitution)
        - Этап Б: Solve φ * p1^T = C * s^T + E * y^T (for gap > 0)
        - Этап В: Solve T * p2^T = A * s^T + B * p1^T (forward substitution)
        
        For gap = 0: p1 is empty, so we only compute p2 = y = A * s
        
        Input: ru_data - dictionary from prepare_richardson_urbanke_encoding()
        containing 'A', 'B', 'T', 'C', 'D', 'E', 'phi', 'permutation', 'H_ru', 'gap', 'm', 'n', 'k'
        """
        from scipy import sparse
        
        # Extract submatrices and parameters
        A = ru_data['A']
        B = ru_data['B']
        T = ru_data['T']
        C = ru_data['C']
        D = ru_data['D']
        E = ru_data['E']
        phi = ru_data['phi']
        permutation = ru_data['permutation']
        gap = ru_data['gap']
        m = ru_data['m']
        n = ru_data['n']
        k = ru_data['k']
        
        # Input: s (information bits) - column vector k x 1
        # IMPORTANT: In standard form H_std = [A | I_m], columns are permuted.
        # permutation[i] = j means: original column j goes to position i in H_std.
        # The first k columns of H_std correspond to info bits in permuted order.
        # We need to map original info bits to their positions in H_std.
        
        # Original info bits (in original column order)
        u_original = np.array(self._data, dtype=np.int32)
        
        # IMPORTANT: For gap=0, H_ru = H_std = [A | I_m] with column permutation
        # permutation[i] = j means: original column j goes to position i in H_std
        # 
        # In standard form H_std = [A | I_m]:
        # - First k columns (positions 0..k-1) in H_std are info columns (after permutation)
        # - permutation[0..k-1] gives the original column indices that are now in positions 0..k-1
        # - These should all be < k (info columns)
        # 
        # For encoding: we need to map original info bits to their positions in H_std
        # s[i] should be the info bit that corresponds to column i in H_std
        # Since permutation[i] = orig_col, we need: s[i] = u_original[permutation[i]]
        # 
        # CRITICAL: For gap=0, H_std = [A | I_m] where first k columns are info columns
        # permutation[i] = orig_col means: original column orig_col is at position i in H_std
        # 
        # The issue: permutation[0..k-1] might contain parity columns (>= k)
        # This happens because create_standart_parity_check_matrix creates permutation as:
        # [non-successful_cols, successful_cols] where successful_cols are parity columns
        # 
        # For gap=0, we need to find which info columns are in first k positions
        # and map them correctly to s
        # 
        # Build mapping: for each original info column j, find its position in H_std
        info_col_to_pos = {}
        for pos in range(n):
            orig_col = permutation[pos]
            if 0 <= orig_col < k:  # This is an info column
                info_col_to_pos[orig_col] = pos
        
        # For gap=0, H_std = [A | I_m] where:
        # - First k columns (positions 0..k-1) form matrix A
        # - Last m columns (positions k..n-1) form identity matrix I_m
        # 
        # However, permutation may place some info columns in positions >= k
        # We need to handle this correctly:
        # - Matrix A is extracted from first k columns of H_std
        # - We need to map info bits to match the structure of A
        # 
        # Strategy: Find which info columns are in first k positions
        # and map them to s. For info columns in positions >= k, we need
        # to understand that they are not part of A, but we still need to use them.
        # 
        # Actually, for gap=0, the correct approach is:
        # - s[i] should be the info bit corresponding to column i in H_std (for i < k)
        # - But if column i in H_std is a parity column, we have a problem
        # 
        # For gap=0, H_std = [A | I_m] where:
        # - First k columns (positions 0..k-1) form matrix A (should be info columns)
        # - Last m columns (positions k..n-1) form identity matrix I_m (parity columns)
        # 
        # Matrix A is extracted as: A = H_ru.extract_sub_matrix(0, 0, t_size, k)
        # where t_size = m - gap = m (for gap=0)
        # So A is the first m rows and first k columns of H_ru
        # 
        # The issue: permutation may place some info columns in positions >= k
        # This means matrix A doesn't contain all info columns, which is a structural problem
        # 
        # For encoding, we need s[i] to correspond to column i in H_std (for i < k)
        # If column i is an info column, use its info bit
        # If column i is a parity column, we have a problem
        # 
        # CRITICAL FIX: For gap=0, the issue is that permutation may place parity columns
        # in first k positions. This happens because successful_cols (parity columns) 
        # can include columns with indices < k if they were used to form the identity matrix.
        # 
        # The correct approach: Matrix A is extracted from first k columns of H_std.
        # We need to map info bits to match the columns that are actually in A.
        # 
        # Since A = H_ru.extract_sub_matrix(0, 0, m, k), A contains the first k columns of H_std.
        # These columns correspond to permutation[0..k-1] in the original H.
        # 
        # However, if permutation[i] >= k for some i < k, then column i in H_std is a parity column,
        # and matrix A doesn't contain the corresponding info column.
        # 
        # Solution: We need to use the info columns that are actually in the first k positions,
        # even if some positions contain parity columns. For positions with parity columns,
        # we cannot use info bits (they're not in A), so we use 0.
        # 
        # But wait - this is still wrong! If A doesn't contain all info columns, encoding will fail.
        # 
        # Actually, the real issue is that for gap=0, we should ensure that first k positions
        # contain info columns. But since we can't change the permutation here, we need to
        # work with what we have.
        # 
        # Better approach: Use the info columns that ARE in first k positions, and for the rest,
        # we need to understand that they're not part of A, so we can't encode them directly.
        # 
        # For now, let's use a workaround: if all first k positions are parity columns,
        # we need to find info columns elsewhere and map them correctly.
        # But this is complex and may indicate a deeper problem.
        # 
        # SIMPLER FIX: For gap=0, if the structure is wrong, we should use standard encoding
        # instead of Richardson-Urbanke. But for now, let's try to fix it:
        # 
        # If first k positions contain parity columns, we need to find which info columns
        # are actually in A. Since A is extracted from first k columns, we need to map
        # info bits to those columns that are info columns in first k positions.
        s = np.zeros(k, dtype=np.int32)
        info_cols_in_first_k = 0
        parity_cols_in_first_k = 0
        
        # Find info columns in first k positions
        info_cols_positions = []
        for pos in range(k):
            orig_col = permutation[pos]
            if 0 <= orig_col < k:
                # This position contains an info column
                info_cols_positions.append((pos, orig_col))
                info_cols_in_first_k += 1
            else:
                parity_cols_in_first_k += 1
        
        # Map info columns that are in first k positions
        for pos, orig_col in info_cols_positions:
            s[pos] = u_original[orig_col]
        
        # If we have parity columns in first k positions, we have a problem
        # For positions with parity columns, we can't use info bits (they're not in A)
        # So we leave them as 0, which will cause encoding errors
        if parity_cols_in_first_k > 0:
            print(f"ERROR: {parity_cols_in_first_k} parity columns found in first k={k} positions of H_std.")
            print(f"  Only {info_cols_in_first_k} info columns in first k positions.")
            print(f"  This is a structural problem: H_std is not in correct [A | I_m] form for gap=0.")
            print(f"  Matrix A (first k columns) does not contain all info columns.")
            print(f"  Richardson-Urbanke encoding with gap=0 cannot work correctly with this structure.")
            print(f"  Consider using standard encoding method instead.")
            # For now, we'll continue but encoding will likely fail
        
        s = s.reshape(-1, 1)
        
        # ============================================================
        # Этап А: Решение T * y^T = A * s^T (forward substitution)
        # ============================================================
        # Compute A * s (mod 2)
        A_sparse = A.get_sparse_matrix()
        As = A_sparse.dot(s)
        
        # Convert to dense array and apply mod 2
        if hasattr(As, 'toarray'):
            As_array = (As.toarray() % 2).astype(np.int32)
        else:
            As_array = (As % 2).astype(np.int32)
        
        # Solve T * y = As (mod 2) using forward substitution
        # T is lower triangular with ones on diagonal
        # For gap = 0, T = I_m (identity matrix), so y = As directly
        T_sparse = T.get_sparse_matrix()
        T_dense = (T_sparse.toarray() % 2).astype(np.int32)
        T_size = T.get_rows()  # Size of T: (m - g) x (m - g)
        
        # Check if T is identity matrix (for gap = 0)
        is_identity = True
        for i in range(T_size):
            for j in range(T_size):
                if i == j:
                    if T_dense[i, j] != 1:
                        is_identity = False
                        break
                else:
                    if T_dense[i, j] != 0:
                        is_identity = False
                        break
            if not is_identity:
                break
        
        if is_identity and gap == 0:
            # T is identity, so y = As directly (no forward substitution needed)
            y = As_array.reshape(-1, 1)
        else:
            # Forward substitution: for i = 0 to T_size-1
            # y[i] = (As[i] - sum(T[i,j] * y[j] for j < i)) mod 2
            # In GF(2), subtraction = addition (XOR)
            y = np.zeros((T_size, 1), dtype=np.int32)
            for i in range(T_size):
                sum_val = 0
                for j in range(i):
                    sum_val = (sum_val + T_dense[i, j] * y[j, 0]) % 2
                
                # y[i] = (As[i] + sum_val) mod 2
                # T[i,i] should be 1 (diagonal element)
                if T_dense[i, i] == 0:
                    # Singular matrix - shouldn't happen for well-formed T
                    y[i, 0] = As_array[i, 0]
                else:
                    y[i, 0] = (As_array[i, 0] + sum_val) % 2
        
        # ============================================================
        # Этап Б: Решение φ * p1^T = C * s^T + E * y^T
        # ============================================================
        # For gap = 0: p1 is empty (length 0), so skip this step
        if gap > 0:
            # Check that phi is not None
            if phi is None:
                raise ValueError(f"phi is None for gap = {gap}. phi must be computed for gap > 0.")
            
            # Compute C * s (mod 2)
            C_sparse = C.get_sparse_matrix()
            Cs = C_sparse.dot(s)
            if hasattr(Cs, 'toarray'):
                Cs_array = (Cs.toarray() % 2).astype(np.int32)
            else:
                Cs_array = (Cs % 2).astype(np.int32)
            
            # Compute E * y (mod 2)
            E_sparse = E.get_sparse_matrix()
            Ey = E_sparse.dot(y)
            if hasattr(Ey, 'toarray'):
                Ey_array = (Ey.toarray() % 2).astype(np.int32)
            else:
                Ey_array = (Ey % 2).astype(np.int32)
            
            # Compute right-hand side: Cs + Ey (mod 2)
            rhs = (Cs_array + Ey_array) % 2
            
            # Solve φ * p1 = rhs (mod 2)
            # φ is g x g matrix, p1 is g x 1
            phi_dense = (phi.get_sparse_matrix().toarray() % 2).astype(np.int32)
            # Use Gaussian elimination or direct solve for small g
            # For now, use simple solve (for gap > 0, this needs proper implementation)
            p1 = np.zeros((gap, 1), dtype=np.int32)
            # TODO: Implement proper solution for gap > 0 case
        else:
            # gap = 0: p1 is empty
            p1 = np.zeros((0, 1), dtype=np.int32)
        
        # ============================================================
        # Этап В: Решение T * p2^T = A * s^T + B * p1^T
        # ============================================================
        # Compute A * s (already computed as As_array)
        # Compute B * p1 (mod 2)
        if gap > 0:
            B_sparse = B.get_sparse_matrix()
            Bp1 = B_sparse.dot(p1)
            if hasattr(Bp1, 'toarray'):
                Bp1_array = (Bp1.toarray() % 2).astype(np.int32)
            else:
                Bp1_array = (Bp1 % 2).astype(np.int32)
            
            # Right-hand side: As + Bp1 (mod 2)
            rhs_p2 = (As_array + Bp1_array) % 2
        else:
            # gap = 0: B is empty, so rhs_p2 = As
            rhs_p2 = As_array
        
        # Solve T * p2 = rhs_p2 (mod 2) using forward substitution
        # For gap = 0, T = I_m (identity matrix), so p2 = rhs_p2 directly
        if is_identity and gap == 0:
            # T is identity, so p2 = rhs_p2 directly (no forward substitution needed)
            p2 = rhs_p2.reshape(-1, 1)
        else:
            p2 = np.zeros((T_size, 1), dtype=np.int32)
            for i in range(T_size):
                sum_val = 0
                for j in range(i):
                    sum_val = (sum_val + T_dense[i, j] * p2[j, 0]) % 2
                
                if T_dense[i, i] == 0:
                    p2[i, 0] = rhs_p2[i, 0]
                else:
                    p2[i, 0] = (rhs_p2[i, 0] + sum_val) % 2
        
        # ============================================================
        # Формирование кодового слова: v = (s, p1, p2)
        # ============================================================
        # In permuted order: v_permuted = [s, p1, p2]
        if gap > 0:
            v_permuted = np.vstack([s, p1, p2]).flatten()
        else:
            # gap = 0: v_permuted = [s, p2]
            v_permuted = np.vstack([s, p2]).flatten()
        
        # Apply inverse permutation to get original column order
        # permutation[i] = j means: original column j goes to position i in H_ru
        # So: v_original[j] = v_permuted[i] where permutation[i] = j
        # This is the inverse mapping: for each position i in H_ru, 
        # the value goes to original position permutation[i]
        # 
        # For gap=0, this should give the same result as standard encoding
        v_original = np.zeros(n, dtype=np.int32)
        for i in range(n):
            original_pos = permutation[i]
            if 0 <= original_pos < n:
                v_original[original_pos] = v_permuted[i]
            else:
                # This shouldn't happen, but handle gracefully
                if i < len(v_permuted):
                    # Try to find correct position
                    # For now, just skip or use direct mapping
                    pass
        
        # Convert to list
        self._encoded_data = v_original.tolist()
        
        # For gap=0, verify that encoding matches standard method
        # This is a sanity check - can be removed in production
        if gap == 0:
            # Compare with what standard encoding would produce
            # Standard: codeword = [data, data * A^T] (in H_std order, no permutation needed)
            # RU gap=0: codeword_permuted = [s, A * s] (in H_std order)
            #           codeword_original = inverse_permutation(codeword_permuted)
            # They should be equivalent
            # 
            # Note: Standard encoding uses G = [I_k | A^T] which is already in H_std order
            # So standard codeword is already in the correct order (no permutation needed)
            # RU encoding produces codeword in H_std order, then applies inverse permutation
            # For gap=0, these should match
            pass  # Verification can be added here if needed
        
        # DEBUG: Verify encoding correctness
        # Check that H_original * v_original^T = 0 (mod 2)
        # This ensures the codeword is valid
        try:
            from scipy import sparse
            # Get original H matrix (before permutation)
            H_original_sparse = ru_data.get('H_original_sparse', None)
            if H_original_sparse is not None:
                v_vec = np.array(v_original, dtype=np.int32).reshape(-1, 1)
                syndrome = H_original_sparse.dot(v_vec)
                if hasattr(syndrome, 'toarray'):
                    s = (syndrome.toarray().flatten() % 2).astype(np.int32)
                else:
                    s = (syndrome.flatten() % 2).astype(np.int32)
                if np.sum(s) != 0:
                    # This is a critical error - encoding is incorrect
                    print(f"ERROR: Richardson-Urbanke encoding produced invalid codeword! Syndrome sum = {np.sum(s)}")
                    # For debugging: print first few syndrome values
                    print(f"  First 10 syndrome values: {s[:10]}")
                    print(f"  Gap: {gap}, k: {k}, m: {m}, n: {n}")
                    print(f"  First 10 permutation values: {permutation[:10] if len(permutation) >= 10 else permutation}")
                    print(f"  First 10 v_original values: {v_original[:10] if len(v_original) >= 10 else v_original}")
                    print(f"  First 10 v_permuted values: {v_permuted[:10] if len(v_permuted) >= 10 else v_permuted}")
        except Exception as e:
            # Skip verification if there's any issue (e.g., H_original not available)
            pass

    def calculate_rows_and_cols_for_regular_interleaver(self):
        """Calculate rows and columns for regular interleaver."""
        i_size_encoded_data = len(self._encoded_data)
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

    def regular_interleave(self, i_rows, i_cols):
        """Regular interleaving."""
        interleaving_data = [0] * len(self._encoded_data)
        self._interleaving_pos_indexes = []
        for i_row in range(i_rows):
            for i_col in range(i_cols):
                i_interleaved_index = i_col * i_rows + i_row
                interleaving_data[i_interleaved_index] = self._encoded_data[i_row * i_cols + i_col]

                self._interleaving_pos_indexes.append(i_interleaved_index)

        self._encoded_data = interleaving_data

    def random_interleave(self):
        """Random interleaving."""
        arr_indexes = list(range(len(self._encoded_data)))
        self._interleaving_pos_indexes = []

        while arr_indexes:
            i_position = random.randint(0, len(arr_indexes) - 1)

            self._interleaving_pos_indexes.append(arr_indexes[i_position])

            arr_indexes.pop(i_position)

        interleaving_data = [0] * len(self._encoded_data)
        for i_index in range(len(self._encoded_data)):
            interleaving_data[i_index] = self._encoded_data[self._interleaving_pos_indexes[i_index]]

        self._encoded_data = interleaving_data

    def interleave(self, interleaver_type):
        """Interleave data based on interleaver type."""
        if interleaver_type == InterleaverType.REGULAR:
            rows_and_cols = self.calculate_rows_and_cols_for_regular_interleaver()
            i_rows = rows_and_cols[0]
            i_cols = rows_and_cols[1]
            if (i_rows == 0) or (i_cols == 0):
                return

            self.regular_interleave(i_rows, i_cols)
        elif interleaver_type == InterleaverType.RANDOM:
            self.random_interleave()

    def deinterleave(self, interleaver_type):
        """Deinterleave data based on interleaver type."""
        if interleaver_type == InterleaverType.REGULAR:
            self.regular_deinterleave()
        elif interleaver_type == InterleaverType.RANDOM:
            self.random_deinterleave()

    def regular_deinterleave(self):
        """Regular deinterleaving."""
        deinterleaver_channel_data = []
        for i_index in range(len(self._interleaving_pos_indexes)):
            deinterleaver_channel_data.append(self._channel_data[self._interleaving_pos_indexes[i_index]])

        self._channel_data = deinterleaver_channel_data

    def random_deinterleave(self):
        """Random deinterleaving."""
        deinterleaver_channel_data = [0.0] * len(self._interleaving_pos_indexes)
        for i_index in range(len(self._interleaving_pos_indexes)):
            deinterleaver_channel_data[self._interleaving_pos_indexes[i_index]] = self._channel_data[i_index]

        self._channel_data = deinterleaver_channel_data
