import numpy as np
from math import lcm

def get_extended_matrix(A, d, r=None, N=None):
    m, n = A.shape
    if not r:
        r = np.ones(n, dtype=int)
    if not N:
        pows = len(str(np.max(np.abs(A)))) + len(str(np.max(np.abs(d)))) + 2
        N = 10 ** pows
    
    rmax = lcm(*r)
    c = np.array([rmax // ri for ri in r])
    
    L = np.zeros((m + n + 1, n + 1), dtype=int)
    
    # First column
    L[:m, 0] = -N * d
    L[m : m + n, 0] = -rmax
    L[m + n, 0] = rmax
    
    # Top-right block
    L[:m, 1:] = N * A
    
    # middle-right block
    for i in range(n):
        L[m + i, 1 + i] = 2 * c[i]
    
    return L, rmax, c

import numpy as np
from fpylll import IntegerMatrix, GSO

def get_gram_schmidt(basis):
    """
    Compute the Gram-Schmidt orthogonalization of the basis vectors.
    
    Parameters:
    -----------
    basis : numpy array of shape (n+1, n-m+1)
        The basis vectors as columns
        
    Returns:
    --------
    b_hat : list of numpy arrays
        The orthogonal vectors from Gram-Schmidt
    mu : numpy array of shape (n-m+1, n-m+1)
        The Gram-Schmidt coefficients where mu[i,j] = <b_i, b_hat_j> / ||b_hat_j||^2
    b_hat_norms_sq : numpy array
        The squared norms of the orthogonal vectors ||b_hat_i||^2
    """
    # Convert basis to fpylll format (basis vectors as rows)
    basis_as_rows = basis.T  # Transpose so vectors are rows
    n_vectors, n_dim = basis_as_rows.shape
    
    # Create IntegerMatrix from basis
    A = IntegerMatrix.from_matrix(basis_as_rows.tolist())
    
    # Create GSO object and update it
    M = GSO.Mat(A)
    M.update_gso()
    
    # Extract the Gram-Schmidt coefficients mu
    # mu[i,j] = <b_i, b_hat_j> / ||b_hat_j||^2 for i > j, and mu[i,i] = 1
    mu = np.zeros((n_vectors, n_vectors))
    for i in range(n_vectors):
        mu[i, i] = 1.0  # Diagonal elements are always 1
        for j in range(i):  # Only for j < i
            mu[i, j] = M.get_mu(i, j)
    
    # Extract squared norms of orthogonal vectors
    # ||b_hat_i||^2 = r[i,i]
    b_hat_norms_sq = np.zeros(n_vectors)
    for i in range(n_vectors):
        b_hat_norms_sq[i] = M.get_r(i, i)
    
    # Compute the orthogonal vectors b_hat
    # We can reconstruct them from the original basis and mu coefficients
    b_hat = []
    for i in range(n_vectors):
        # b_hat[i] = b[i] - sum_{j<i} mu[i,j] * b_hat[j]
        b_hat_i = basis[:, i].astype(float)
        for j in range(i):
            b_hat_i = b_hat_i - mu[i, j] * b_hat[j]
        b_hat.append(b_hat_i)
    
    return b_hat, mu, b_hat_norms_sq

# Test with your actual basis
def test_gram_schmidt():
    """Test the Gram-Schmidt function with your data"""
    # First get the basis from your previous steps
    m, n = 3, 20
    A = np.array(
    [
      [46,  8, 29, 40, 28, 11, 34, 31, 31, 50, 13, 40, 27, 44, 25, 27, 16, 43, 45, 10],
      [40, 15, 23, 38, 15, 36, 46,  1, 26, 27, 23,  1,  6, 45,  7, 42, 10, 17, 10,  7],
      [16,  6, 10, 31,  0, 32, 22, 43, 47, 38, 30, 45,  5, 23, 12, 22, 26, 47, 15, 45]
    ], dtype=int
    )
    d = np.array([299, 217, 257], dtype=int)
    
    # Get extended matrix
    L, rmax, c = get_extended_matrix(A, d)
    
    # Get reduced basis
    basis = get_reduced_basis(m, n, L)
    
    print(f"Basis shape: {basis.shape}")
    print(f"Expected: ({n+1}, {n-m+1})")
    
    # Compute Gram-Schmidt
    b_hat, mu, b_hat_norms_sq = get_gram_schmidt(basis)
    
    print(f"\nNumber of orthogonal vectors: {len(b_hat)}")
    print(f"Shape of mu matrix: {mu.shape}")
    print(f"Number of squared norms: {len(b_hat_norms_sq)}")
    
    # Verify orthogonality of b_hat vectors
    print("\nOrthogonality check (should be close to 0):")
    for i in range(min(3, len(b_hat))):
        for j in range(i):
            dot_product = np.dot(b_hat[i], b_hat[j])
            print(f"<b_hat[{i}], b_hat[{j}]> = {dot_product:.2e}")
    
    # Verify norms
    print("\nNorm verification:")
    for i in range(min(3, len(b_hat))):
        computed_norm_sq = np.dot(b_hat[i], b_hat[i])
        stored_norm_sq = b_hat_norms_sq[i]
        print(f"||b_hat[{i}]||^2: computed={computed_norm_sq:.2f}, stored={stored_norm_sq:.2f}")
    
    return b_hat, mu, b_hat_norms_sq

if __name__ == "__main__":
    b_hat, mu, b_hat_norms_sq = test_gram_schmidt()


# m, n = 3, 20
# A = np.array(
# [
#   [46,  8, 29, 40, 28, 11, 34, 31, 31, 50, 13, 40, 27, 44, 25, 27, 16, 43, 45, 10],
#   [40, 15, 23, 38, 15, 36, 46,  1, 26, 27, 23,  1,  6, 45,  7, 42, 10, 17, 10,  7],
#   [16,  6, 10, 31,  0, 32, 22, 43, 47, 38, 30, 45,  5, 23, 12, 22, 26, 47, 15, 45]
# ], dtype=int
# )
# d = np.array([299, 217, 257], dtype=int)

# L, rmax, c = get_extended_matrix(A, d)
# print(L)
# print(rmax)
# print(c)

# np.savetxt('ext_mat.txt', L, fmt='%d', delimiter = ' ')

