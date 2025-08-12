import numpy as np
from scipy.linalg import null_space
import fpylll

def extract_complete_problem_data():
    """Extract the complete coefficient matrix including x1, x3, x5, x7, x9, x11, x13"""
    
    # The constraint matrix is actually [I | A_binary]
    # where I is 7x7 identity for x1, x3, x5, x7, x9, x11, x13
    # and A_binary is the 7x60 matrix for x15-x74
    
    # Identity part for x1, x3, x5, x7, x9, x11, x13
    I = np.eye(7, dtype=int)
    
    # Binary variables part (x15-x74)
    A_binary = np.array([
        [74, 49, 12, 93, 56, 16, 39, 77, 56, 73, 1, 3, 68, 61, 8, 55, 18, 21, 57, 98, 58, 57, 46, 72, 6, 16, 76, 21, 78, 18, 11, 58, 59, 25, 32, 14, 16, 3, 60, 12, 7, 42, 98, 34, 33, 16, 97, 63, 66, 28, 57, 19, 74, 44, 45, 49, 76, 74, 9, 44],
        [20, 7, 68, 69, 95, 64, 76, 12, 45, 43, 83, 15, 90, 10, 96, 98, 53, 1, 2, 58, 24, 90, 29, 57, 19, 73, 89, 31, 12, 34, 67, 48, 11, 22, 36, 78, 75, 52, 95, 57, 62, 94, 10, 42, 89, 11, 77, 85, 30, 82, 20, 52, 78, 6, 57, 65, 79, 83, 16, 67],
        [85, 47, 67, 59, 84, 59, 19, 8, 50, 66, 5, 51, 51, 64, 64, 53, 61, 45, 3, 76, 17, 54, 13, 89, 68, 57, 4, 24, 96, 81, 36, 54, 3, 82, 33, 88, 1, 29, 4, 48, 51, 14, 86, 64, 73, 78, 45, 65, 30, 52, 6, 78, 9, 19, 87, 73, 10, 87, 33, 1],
        [13, 71, 78, 84, 56, 66, 8, 68, 48, 28, 33, 34, 8, 99, 80, 74, 2, 10, 96, 41, 98, 74, 39, 91, 85, 95, 96, 1, 80, 90, 97, 36, 7, 69, 9, 9, 93, 94, 44, 36, 71, 37, 72, 38, 74, 89, 37, 24, 88, 77, 61, 80, 2, 60, 87, 80, 74, 42, 2, 37],
        [35, 61, 66, 78, 46, 89, 61, 25, 55, 16, 81, 35, 96, 23, 83, 39, 14, 53, 23, 23, 93, 38, 15, 20, 19, 28, 79, 51, 24, 6, 3, 47, 61, 60, 71, 63, 26, 66, 71, 63, 56, 32, 39, 31, 64, 89, 62, 68, 59, 71, 48, 76, 96, 74, 61, 21, 46, 18, 23, 24],
        [86, 8, 44, 96, 64, 65, 68, 53, 19, 33, 28, 42, 72, 39, 5, 77, 37, 89, 7, 78, 10, 78, 10, 96, 55, 1, 64, 61, 63, 90, 22, 78, 92, 25, 24, 65, 6, 68, 66, 66, 1, 67, 78, 21, 47, 17, 89, 77, 88, 54, 10, 87, 88, 80, 76, 9, 83, 95, 86, 24],
        [41, 64, 82, 24, 48, 41, 29, 93, 64, 39, 92, 86, 64, 45, 87, 34, 39, 88, 99, 63, 85, 48, 83, 88, 85, 5, 14, 31, 12, 93, 55, 1, 2, 22, 93, 49, 35, 25, 39, 1, 77, 43, 7, 42, 36, 63, 5, 8, 43, 18, 60, 47, 47, 46, 45, 38, 9, 37, 8, 82]
    ])
    
    # Complete coefficient matrix: [I | A_binary]
    A_complete = np.hstack([I, A_binary])
    
    b = np.array([1324, 1554, 1429, 1686, 1482, 1613, 1424])
    
    # Variable structure:
    # x1, x3, x5, x7, x9, x11, x13 (7 continuous/integer variables)
    # x15-x74 (60 binary variables)
    # Total: 67 variables
    
    return A_complete, b

def compute_kernel_basis_hermite(A):
    """Compute integer kernel basis using a more robust method"""
    m, n = A.shape
    
    # Method 1: Using nullspace computation
    from sympy import Matrix
    A_sympy = Matrix(A)
    kernel_basis = A_sympy.nullspace()
    
    if len(kernel_basis) > 0:
        # Convert to numpy array
        Q = np.array([list(vec) for vec in kernel_basis]).T
    else:
        # Fallback to scipy
        Q = null_space(A)
        # Scale and round to get integer basis
        scale = 10000
        Q = np.round(Q * scale).astype(int)
        for i in range(Q.shape[1]):
            g = np.gcd.reduce(Q[:, i])
            if g > 0:
                Q[:, i] = Q[:, i] // g
    
    return Q

def apply_lll_with_preprocessing(Q):
    """Apply LLL reduction with preprocessing as mentioned in the paper"""
    from fpylll import IntegerMatrix, LLL
    
    n, k = Q.shape
    
    # According to the paper, they start with a basis of specific structure
    # (upper triangular form as in Lemma 1)
    
    # Convert to fpylll format
    basis_list = Q.T.tolist()
    M = IntegerMatrix.from_matrix(basis_list)
    
    # Apply LLL with specific reduction factor
    # Paper uses γ = 0.95 in experiments
    LLL.reduction(M, delta=0.95)
    
    # Convert back
    Q_reduced = np.array(M).T
    
    # Apply column operations to get identity in bottom part
    # as discussed in Section 4 of the paper
    Q_reduced = apply_column_operations(Q_reduced)
    
    return Q_reduced

def apply_column_operations(Q):
    """Apply elementary column operations to achieve sparse structure"""
    n, k = Q.shape
    Q_copy = Q.copy()
    
    # Start from the last column and work backwards
    # Try to create identity structure in the bottom part
    
    # Find the transition point where sparsity starts
    dense_rows = 0
    for i in range(n):
        row = Q_copy[i, :]
        if np.count_nonzero(row) > k // 2:  # If more than half non-zero
            dense_rows += 1
        else:
            break
    
    # For the sparse part, try to make it identity-like
    if dense_rows < n:
        sparse_start = dense_rows
        sparse_cols = min(k, n - sparse_start)
        
        # Apply column operations to create identity in bottom-right
        for j in range(sparse_cols):
            col_idx = k - sparse_cols + j
            row_idx = sparse_start + j
            
            if row_idx < n and col_idx < k:
                # Make this position ±1 and others in column 0
                if Q_copy[row_idx, col_idx] != 0:
                    # Scale column if needed
                    factor = Q_copy[row_idx, col_idx]
                    if abs(factor) != 1:
                        Q_copy[:, col_idx] = Q_copy[:, col_idx] // factor
                    
                    # Eliminate other entries in this part of the column
                    for i in range(sparse_start, n):
                        if i != row_idx and Q_copy[i, col_idx] != 0:
                            mult = Q_copy[i, col_idx] // Q_copy[row_idx, col_idx]
                            Q_copy[:, col_idx] -= mult * Q_copy[:, row_idx]
    
    return Q_copy

def analyze_complete_structure(Q, A_complete):
    """Analyze the structure with respect to variable types"""
    n, k = Q.shape
    
    print(f"Kernel basis dimensions: {n} x {k}")
    print(f"Expected: 67 x 60 (since rank(A) = 7)")
    
    # Separate analysis for continuous and binary variables
    continuous_part = Q[:7, :]  # Rows for x1, x3, x5, x7, x9, x11, x13
    binary_part = Q[7:, :]       # Rows for x15-x74
    
    print(f"\nContinuous variables part shape: {continuous_part.shape}")
    print(f"Binary variables part shape: {binary_part.shape}")
    
    # Count dense rows
    dense_count = 0
    for i in range(n-1, -1, -1):
        row = Q[i, :]
        non_zero = np.count_nonzero(row)
        if non_zero == 1 and abs(row[np.nonzero(row)[0][0]]) == 1:
            continue
        else:
            dense_count = n - i
            break
    
    print(f"\nDense rows (from top): {dense_count}")
    print(f"Sparse identity-like rows (from bottom): {n - dense_count}")
    
    # Check if we have the desired structure
    if dense_count < n:
        print(f"\nStructure achieved: X1 (dense) | X2 (mixed)")
        print(f"                    0         | I  (sparse)")
    
    return dense_count

# Updated main function
def lattice_reformulation_complete():
    """Complete lattice reformulation for markshare2"""
    
    print("=== Markshare2 Lattice Reformulation ===\n")
    
    # Get complete problem data
    A_complete, b = extract_complete_problem_data()
    print(f"Complete constraint matrix shape: {A_complete.shape}")
    print(f"Variables: 7 continuous (x1,x3,x5,x7,x9,x11,x13) + 60 binary (x15-x74)")
    
    # Compute kernel basis
    print("\nComputing kernel basis...")
    Q = compute_kernel_basis_hermite(A_complete)
    print(f"Initial kernel basis shape: {Q.shape}")
    
    # Apply LLL reduction
    print("\nApplying LLL reduction...")
    Q_reduced = apply_lll_with_preprocessing(Q)
    
    # Analyze structure
    print("\nAnalyzing reduced basis structure...")
    dense_rows = analyze_complete_structure(Q_reduced, A_complete)
    
    return Q_reduced, dense_rows

if __name__ == "__main__":
    Q_reduced, dense_rows = lattice_reformulation_complete()
    
    # According to Table 5 in the paper, for m=7 (6x50 is close):
    # They observe around 46-49 dense rows
    print(f"\nExpected dense rows (from paper): ~46-49")
    print(f"Observed dense rows: {dense_rows}")
