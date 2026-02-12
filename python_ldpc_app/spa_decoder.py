import math

try:
    import numpy as np
    from scipy import sparse
except ImportError as e:
    raise ImportError(
        "numpy and scipy are required but not installed. "
        "Please install them using: pip install numpy scipy"
    ) from e

from enums import Result


class SPA_Decoder:
    def __init__(self, encoder_decoder_data, settings):
        self.m_pData = encoder_decoder_data
        self.m_pSettings = settings
        self._arr_changed_by_iterations = []
        self._normalized_llr_by_iterations = []
        self._normalized_llr_by_iterations_soft = []
        self._d_summarize_normalized_llr = 0.0
        self._arr_aposteriori_llrs = []
        self.convergence_iteration = -1
        
        # Get sparse matrix H for efficient access (cached reference)
        # Use cached H_sparse if available, otherwise get it
        if hasattr(encoder_decoder_data, '_h_sparse_cached'):
            self.H_sparse = encoder_decoder_data._h_sparse_cached
        else:
            self.H_sparse = self.m_pData._h_std.get_sparse_matrix()
        
        # Use cached decoder structures if available (for multi-processing efficiency)
        # These structures come from encoder_decoder_data which is shared via pickle
        # Otherwise, compute them (for backward compatibility)
        if hasattr(encoder_decoder_data, '_decoder_structures_initialized') and encoder_decoder_data._decoder_structures_initialized:
            # Use cached structures
            self.H_coo, self.var_to_check, self.check_to_var = encoder_decoder_data.get_decoder_structures()
        else:
            # Compute structures (backward compatibility for single-threaded mode)
            self.H_coo = self.H_sparse.tocoo()
            self._init_neighbor_structures()
    
    def _init_neighbor_structures(self):
        """Precompute neighbor structures for efficient iteration using sparse matrix.
        This is only called if cached structures are not available.
        """
        # Variable nodes -> check nodes
        self.var_to_check = {}
        # Check nodes -> variable nodes
        self.check_to_var = {}
        
        # Build neighbor structures from COO format
        for i, j in zip(self.H_coo.row, self.H_coo.col):
            if j not in self.var_to_check:
                self.var_to_check[j] = []
            self.var_to_check[j].append(i)
            
            if i not in self.check_to_var:
                self.check_to_var[i] = []
            self.check_to_var[i].append(j)
    
    def decode(self, p_data_buffer):
        """Decode data using Sum-Product Algorithm - C++ implementation with sparse matrices."""
        self.convergence_iteration = -1
        num_check_nodes = self.m_pData._m
        num_variable_nodes = self.m_pData._n
        
        # Initialize z (hard decision vector) as numpy array
        z = np.zeros(num_variable_nodes, dtype=np.int32)
        
        # Get H matrix data (sparse representation)
        # Use sparse matrix directly for efficient access
        
        # Matrix of a priori data: M[check_node][variable_node]
        # Use sparse matrix (LIL format for efficient element-wise access, then convert to CSR)
        # M has same sparsity pattern as H
        M = sparse.lil_matrix((num_check_nodes, num_variable_nodes), dtype=np.float64)
        
        # Matrix of a posteriori data: E[check_node][variable_node]
        # Use sparse matrix (LIL format for efficient element-wise access, then convert to CSR)
        # E has same sparsity pattern as H
        E = sparse.lil_matrix((num_check_nodes, num_variable_nodes), dtype=np.float64)
        
        # Initialize a priori matrix M using sparse matrix operations
        # M[iCheckNode][iVariableNode] = H[iCheckNode][iVariableNode] * channelData[iVariableNode]
        # Use COO format for efficient initialization
        channel_data_array = np.array(p_data_buffer._channel_data, dtype=np.float64)
        for i, j in zip(self.H_coo.row, self.H_coo.col):
            M[i, j] = channel_data_array[j]
        
        # Keep M in LIL format (will convert to CSR only when needed for row access)
        
        # Initialize arrays for LLR tracking (numpy arrays for vectorized operations)
        arr_apriori_llrs = channel_data_array.copy()
        arr_aposteriori_llrs = np.zeros(num_variable_nodes, dtype=np.float64)
        
        i_cur_iter = 0
        b_is_finished = False
        
        # Main iteration loop: while (!bIsFinished || (iCurIter < maxIterations))
        # OPTIMIZATION: Keep E in LIL format throughout iteration to avoid conversions
        # Only convert to CSR/CSC when needed for operations
        while (not b_is_finished) or (i_cur_iter < self.m_pSettings.get_max_iterations()):
            # Compute extrinsic message from check node j to bit node i
            # E[iCheckNode][iVariableNode] = 2 * atanh(prod(tanh(M[iCheckNode][iVariableNode2]/2) for all iVariableNode2 != iVariableNode))
            # OPTIMIZATION: Use "leave-one-out" method with sparse matrices
            # Keep E in LIL format for efficient element-wise updates (avoid conversions)
            
            # Get M row data efficiently
            # Convert M to CSR only once per iteration for efficient row access
            M_csr = M.tocsr() if not isinstance(M, sparse.csr_matrix) else M
            
            for i_check_node in range(num_check_nodes):
                if i_check_node not in self.check_to_var:
                    continue
                
                var_nodes = self.check_to_var[i_check_node]
                num_var_nodes = len(var_nodes)
                
                if num_var_nodes == 0:
                    continue
                
                # Get M row data efficiently using CSR format
                row_start = M_csr.indptr[i_check_node]
                row_end = M_csr.indptr[i_check_node + 1]
                M_row_indices = M_csr.indices[row_start:row_end]
                M_row_data = M_csr.data[row_start:row_end]
                
                # Create mapping from variable node to M value (O(degree) lookup)
                var_to_m = {M_row_indices[idx]: M_row_data[idx] for idx in range(len(M_row_indices))}
                
                # Pre-compute all tanh(M/2) values for this check node (vectorized)
                tanh_values = []
                for i_variable_node in var_nodes:
                    # Get M value from mapping (much faster than M[i, j] access for LIL)
                    m_val = var_to_m.get(i_variable_node, 0.0)
                    d_temp = m_val / 2.0
                    # Clip tanh input for numerical stability
                    if d_temp > 17.5:
                        tanh_val = 0.99999999999999878
                    elif d_temp < -17.5:
                        tanh_val = -0.99999999999999878
                    else:
                        tanh_val = np.tanh(d_temp)
                    tanh_values.append(tanh_val)
                
                # Compute product of all tanh values (once per check node)
                if tanh_values:
                    # Use numpy for efficient product computation
                    tanh_array = np.array(tanh_values, dtype=np.float64)
                    total_product = np.prod(tanh_array)
                    
                    # Compute E for each variable node using "leave-one-out"
                    # E = 2 * atanh(total_product / tanh_value)
                    # Vectorized computation for better performance
                    for idx, i_variable_node in enumerate(var_nodes):
                        tanh_val = tanh_values[idx]
                        if abs(tanh_val) > 1e-10:  # Avoid division by very small numbers
                            # Leave-one-out product (O(1) instead of O(degree))
                            d_result = total_product / tanh_val
                        else:
                            # If tanh_value is very small, compute product without it
                            d_result = np.prod(np.delete(tanh_array, idx))
                        
                        # Clip atanh input for numerical stability
                        d_result_clipped = np.clip(d_result, -0.99999999999999878, 0.99999999999999878)
                        E[i_check_node, i_variable_node] = 2.0 * np.arctanh(d_result_clipped)
            
            # Test: compute a posteriori LLRs and hard decisions
            # OPTIMIZATION: Use sparse matrix operations for summing E messages
            # Initialize with channel data
            arr_aposteriori_llrs = channel_data_array.copy()
            
            # Sum all E messages from check nodes using sparse matrix operations
            # Convert E to CSC format only once for efficient column operations
            E_csc = E.tocsc() if not isinstance(E, sparse.csc_matrix) else E
            
            # Vectorized column sum: E^T * ones gives sum of E messages for each variable node
            # This is much faster than iterating over columns
            ones_vector = np.ones(num_check_nodes, dtype=np.float64)
            E_sum = E_csc.transpose().dot(ones_vector)
            
            # Add E sums to channel data (vectorized)
            arr_aposteriori_llrs += E_sum
            
            # Hard decision: z[i] = 1 if L < 0, else 0 (vectorized)
            z = (arr_aposteriori_llrs < 0.0).astype(np.int32)
            
            # Compute z_reversed = z ^ 1 (vectorized for better performance)
            z_array = np.array(z, dtype=np.int32)
            z_reversed_array = (z_array ^ 1).reshape(-1, 1)
            
            # Calculate syndrome vector: s = H * z_reversed (mod 2)
            result = self.H_sparse.dot(z_reversed_array)
            
            # Apply mod 2 and check if all zeros (vectorized)
            if hasattr(result, 'toarray'):
                s = result.toarray().flatten() % 2
            else:
                s = result.flatten() % 2
            
            # Check if syndrome is all zeros (vectorized - much faster)
            b_is_finished = (np.sum(s) == 0)
            
            # Normalized LLR calculation
            # This metric counts the number of sign changes between a priori and a posteriori LLRs
            # for information bits (first k bits) where |a posteriori LLR| <= 1.0
            # Normalized LLR = (number of sign changes) / k
            if self.m_pSettings.is_normalized_llr_calculate():
                i_cnt_changes_la_le = 0
                k = num_variable_nodes - num_check_nodes
                
                # Check only information bits (first k bits)
                for i_index in range(k):
                    # Skip if a posteriori LLR magnitude is too large (> 1.0)
                    # This filters out bits with high confidence
                    if abs(arr_aposteriori_llrs[i_index]) > 7.0:
                        continue
                    
                    # Count sign changes: a priori and a posteriori have opposite signs
                    # This indicates uncertainty or change in belief
                    if arr_apriori_llrs[i_index] * arr_aposteriori_llrs[i_index] < 0.0:
                        i_cnt_changes_la_le += 1
                
                self._arr_changed_by_iterations.append(i_cnt_changes_la_le)
                # Normalize by number of information bits
                self._normalized_llr_by_iterations.append(float(i_cnt_changes_la_le) / k if k > 0 else 0.0)
            
            # Early termination if syndrome is zero
            if b_is_finished:
                self.convergence_iteration = i_cur_iter
                p_data_buffer._decoded_data = z.tolist()
                
                # Use the last normalized LLR value from the current iteration
                # (matches C++ implementation: _dSummarizeNormalizedLLR = _normalizedLLRByIterations.at(_normalizedLLRByIterations.size() - 1))
                if self.m_pSettings.is_normalized_llr_calculate():
                    if self._normalized_llr_by_iterations:
                        self._d_summarize_normalized_llr = self._normalized_llr_by_iterations[-1]
                
                return Result.OK
            
            # Check if max iterations reached
            if i_cur_iter == (self.m_pSettings.get_max_iterations() - 1):
                p_data_buffer._decoded_data = z.tolist()
                
                # Use the last normalized LLR value from the current iteration
                # (matches C++ implementation: _dSummarizeNormalizedLLR = _normalizedLLRByIterations.at(_normalizedLLRByIterations.size() - 1))
                if self.m_pSettings.is_normalized_llr_calculate():
                    if self._normalized_llr_by_iterations:
                        self._d_summarize_normalized_llr = self._normalized_llr_by_iterations[-1]
                
                return Result.DATA_TRANSFER_NOT_OK
            
            # Update M matrix for next iteration
            # M[iCheckNode][iVariableNode] = channelData[iVariableNode] + sum(E[iCheckNode2][iVariableNode] for all iCheckNode2 != iCheckNode)
            # OPTIMIZATION: Use sparse matrix operations
            # M = (arr_aposteriori_llrs broadcast to H pattern) - E
            # Keep M in LIL format for efficient element-wise updates (avoid conversions)
            M = M.tolil() if not isinstance(M, sparse.lil_matrix) else M
            
            # Get E values efficiently (E is in LIL format)
            # For each edge (i, j) in H, set M[i, j] = arr_aposteriori_llrs[j] - E[i, j]
            for i, j in zip(self.H_coo.row, self.H_coo.col):
                # M = total - E from this check node (extrinsic message)
                # Direct LIL access is efficient
                e_val = E[i, j] if i < E.shape[0] and j < E.shape[1] else 0.0
                M[i, j] = arr_aposteriori_llrs[j] - e_val
            
            # Keep M in LIL format for next iteration (will convert to CSR only when needed)
            
            # Update a priori LLRs for next iteration (for normalized LLR)
            if self.m_pSettings.is_normalized_llr_calculate():
                arr_apriori_llrs = arr_aposteriori_llrs.copy()
            
            i_cur_iter += 1
        
        # Maximum iterations reached (should not reach here due to check above, but for safety)
        p_data_buffer._decoded_data = z.tolist()
        return Result.DATA_TRANSFER_NOT_OK
