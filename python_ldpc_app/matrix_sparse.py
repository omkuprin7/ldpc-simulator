"""
Matrix class using scipy.sparse for efficient sparse matrix operations.
Optimized for LDPC codes where matrices are typically sparse.
"""
try:
    import numpy as np
    from scipy import sparse
except ImportError as e:
    raise ImportError(
        "numpy and scipy are required but not installed. "
        "Please install them using: pip install numpy scipy"
    ) from e


class SparseMatrix:
    """Sparse matrix class using scipy.sparse for LDPC operations."""
    
    def __init__(self, rows=0, cols=0, default_value=0, values=None, sparse_matrix=None):
        """
        Initialize sparse matrix.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            default_value: Default value for elements (usually 0)
            values: List of lists (dense format) - will be converted to sparse
            sparse_matrix: scipy.sparse matrix (direct assignment)
        """
        if sparse_matrix is not None:
            self._matrix = sparse_matrix
            self._rows = sparse_matrix.shape[0]
            self._cols = sparse_matrix.shape[1]
        elif values is not None:
            # Convert dense matrix to sparse
            dense = np.array(values, dtype=np.int32)
            self._matrix = sparse.csr_matrix(dense)
            self._rows = len(values)
            self._cols = len(values[0]) if values else 0
        else:
            self._rows = rows
            self._cols = cols
            if default_value == 0:
                # Create empty sparse matrix
                self._matrix = sparse.csr_matrix((rows, cols), dtype=np.int32)
            else:
                # Create dense matrix with default value, then convert to sparse
                dense = np.full((rows, cols), default_value, dtype=np.int32)
                self._matrix = sparse.csr_matrix(dense)
    
    def get_rows(self):
        """Get the number of rows (M). M = N - K. Parity bits."""
        return self._rows
    
    def get_cols(self):
        """Get the number of columns (N)."""
        return self._cols
    
    def get_message_bit_length(self):
        """Get the message length (K). K = N - M"""
        return self._cols - self._rows
    
    def get_data(self):
        """Get matrix data as list of lists (for compatibility)."""
        return self._matrix.toarray().tolist()
    
    def get_sparse_matrix(self):
        """Get underlying scipy sparse matrix."""
        return self._matrix
    
    def update_from_sparse_matrix(self, sparse_matrix):
        """Update internal matrix from a scipy sparse matrix.
        Used for optimization when modifying matrix structure.
        """
        self._matrix = sparse_matrix
    
    def set_element(self, i_row, i_col, i_value):
        """Set an element at a specific position (iRow, iCol)."""
        if i_row >= self._rows or i_col >= self._cols or i_row < 0 or i_col < 0:
            raise IndexError("Index out of bounds")
        
        # For sparse matrices, setting elements one by one is inefficient
        # But for Gaussian elimination we need this functionality
        # Convert to LIL format which is more efficient for element-wise modifications
        lil = self._matrix.tolil()
        lil[i_row, i_col] = i_value
        # Convert back to CSR
        self._matrix = lil.tocsr()
    
    def get_element(self, i_row, i_col):
        """Get an element at a specific position (iRow, iCol)."""
        if i_row >= self._rows or i_col >= self._cols or i_row < 0 or i_col < 0:
            raise IndexError("Index out of bounds")
        
        return int(self._matrix[i_row, i_col])
    
    def multiply(self, mtx):
        """Multiply two matrices (mod 2) in GF(2) arithmetic."""
        if self._cols != mtx.get_rows():
            raise ValueError(
                "Number of columns of the first matrix must be equal to the number of rows of the second matrix."
            )
        
        # Matrix multiplication using sparse matrices
        # In GF(2), we need to compute dot product mod 2 for each element
        result = self._matrix.dot(mtx.get_sparse_matrix())
        
        # Convert to COO format for efficient mod 2 operation
        coo = result.tocoo()
        
        # Apply mod 2 to all values
        # In GF(2), any value mod 2 gives 0 or 1
        coo.data = coo.data % 2
        
        # Remove zeros (elements that are 0 mod 2)
        # This is important because after mod 2, many elements become 0
        coo.eliminate_zeros()
        
        # Convert back to CSR format
        result = coo.tocsr()
        
        return SparseMatrix(sparse_matrix=result)
    
    def transpose(self):
        """Transpose the matrix."""
        transposed = self._matrix.transpose()
        return SparseMatrix(sparse_matrix=transposed)
    
    def permute_columns(self, permutation):
        """Permute columns of the matrix.
        permutation[iCol] gives the old column index that should be at position iCol.
        In C++: permutedMatrix[iRow][iCol] = _data[iRow][permutation[iCol]]
        This means: new column iCol comes from old column permutation[iCol]
        
        Example: if permutation = [2, 0, 1], then:
        - new column 0 comes from old column 2
        - new column 1 comes from old column 0
        - new column 2 comes from old column 1
        """
        if len(permutation) != self._cols:
            raise ValueError("Invalid permutation size")
        
        # Convert to COO for efficient column permutation
        coo = self._matrix.tocoo()
        
        # Create mapping: for each old column index, find its new position
        # permutation[new_pos] = old_col means: old column old_col goes to position new_pos
        # So we need: for old_col, find new_pos where permutation[new_pos] == old_col
        old_to_new = {}
        for new_pos, old_col in enumerate(permutation):
            old_to_new[old_col] = new_pos
        
        # Map old column indices to new column indices
        new_cols = np.array([old_to_new[col] for col in coo.col], dtype=coo.col.dtype)
        
        # Create new COO matrix with permuted columns
        new_coo = sparse.coo_matrix(
            (coo.data, (coo.row, new_cols)),
            shape=(self._rows, self._cols),
            dtype=coo.dtype
        )
        
        # Convert back to CSR
        self._matrix = new_coo.tocsr()
    
    def swap_rows(self, row1, row2):
        """Swap two rows in the matrix."""
        if row1 >= self._rows or row2 >= self._rows or row1 < 0 or row2 < 0:
            raise IndexError("Row index out of bounds")
        
        # Convert to COO for row swapping
        coo = self._matrix.tocoo()
        
        # Swap row indices
        mask1 = coo.row == row1
        mask2 = coo.row == row2
        coo.row[mask1] = row2
        coo.row[mask2] = row1
        
        # Convert back to CSR
        self._matrix = coo.tocsr()
    
    def permute_rows(self, permutation):
        """Permute rows of the matrix.
        permutation[iRow] gives the old row index that should be at position iRow.
        Similar to permute_columns but for rows.
        """
        if len(permutation) != self._rows:
            raise ValueError("Invalid permutation size")
        
        # Convert to COO for efficient row permutation
        coo = self._matrix.tocoo()
        
        # Create mapping: for each old row index, find its new position
        old_to_new = {}
        for new_pos, old_row in enumerate(permutation):
            old_to_new[old_row] = new_pos
        
        # Map old row indices to new row indices
        new_rows = np.array([old_to_new[row] for row in coo.row], dtype=coo.row.dtype)
        
        # Create new COO matrix with permuted rows
        new_coo = sparse.coo_matrix(
            (coo.data, (new_rows, coo.col)),
            shape=(self._rows, self._cols),
            dtype=coo.dtype
        )
        
        # Convert back to CSR
        self._matrix = new_coo.tocsr()
    
    def extract_sub_matrix(self, start_row, start_col, sub_rows, sub_cols):
        """Extract a sub-matrix."""
        if start_row < 0 or start_col < 0 or start_row + sub_rows > self._rows or start_col + sub_cols > self._cols:
            raise IndexError("Sub-matrix dimensions are out of bounds")
        
        # Convert to CSR format for efficient slicing (if not already CSR)
        if not isinstance(self._matrix, sparse.csr_matrix):
            matrix_csr = self._matrix.tocsr()
        else:
            matrix_csr = self._matrix
        
        sub_matrix = matrix_csr[start_row:start_row + sub_rows, start_col:start_col + sub_cols]
        return SparseMatrix(sparse_matrix=sub_matrix)
    
    @staticmethod
    def create_identity_matrix(size):
        """Create an identity matrix."""
        identity = sparse.eye(size, format='csr', dtype=np.int32)
        return SparseMatrix(sparse_matrix=identity)
    
    def concatenate_horizontally(self, mtx):
        """Concatenate matrices horizontally."""
        if self._rows != mtx.get_rows():
            raise ValueError("Row counts must match for horizontal concatenation")
        
        result = sparse.hstack([self._matrix, mtx.get_sparse_matrix()])
        return SparseMatrix(sparse_matrix=result)
    
    def concatenate_vertically(self, mtx):
        """Concatenate matrices vertically."""
        if self._cols != mtx.get_cols():
            raise ValueError("Column counts must match for vertical concatenation")
        
        result = sparse.vstack([self._matrix, mtx.get_sparse_matrix()])
        return SparseMatrix(sparse_matrix=result)
    
    def add_row(self, row):
        """Add a row to the matrix."""
        if self._cols != len(row):
            raise ValueError("Must have the same dimensions for addition.")
        
        row_matrix = sparse.csr_matrix([row], dtype=np.int32)
        self._matrix = sparse.vstack([self._matrix, row_matrix])
        self._rows += 1
    
    def print(self):
        """Print the matrix (converts to dense for display)."""
        dense = self._matrix.toarray()
        for row in dense:
            print(" ".join(str(int(value)) for value in row))
    
    def init(self, rows, cols, default_value=0):
        """Initialize the matrix."""
        self._rows = rows
        self._cols = cols
        if default_value == 0:
            self._matrix = sparse.csr_matrix((rows, cols), dtype=np.int32)
        else:
            dense = np.full((rows, cols), default_value, dtype=np.int32)
            self._matrix = sparse.csr_matrix(dense)
