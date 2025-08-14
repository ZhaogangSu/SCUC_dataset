import numpy as np
from math import lcm
from fpylll import IntegerMatrix, LLL, GSO


class MarketSplit:
   def __init__(self, A, d, r=None):
       self.A = A
       self.d = d
       self.m, self.n = A.shape
       self.r = r if r is not None else np.ones(self.n, dtype=int)
       
       # Computed attributes
       self.L = None
       self.rmax = None
       self.c = None
       self.basis = None
       self.b_hat = None
       self.mu = None
       self.b_hat_norms_sq = None
       self.b_bar = None  # Store dual basis
       self.b_bar_norms = None
       
       # Run preprocessing
       self._get_extended_matrix()
       self._get_reduced_basis()
       self._get_gso()
       self._compute_dual_norms()
   
   def _get_extended_matrix(self, N=None):
       if N is None:
           pows = len(str(np.max(np.abs(self.A)))) + len(str(np.max(np.abs(self.d)))) + 2
           N = 10 ** pows
       
       self.rmax = lcm(*self.r)
       self.c = np.array([self.rmax // ri for ri in self.r])
       
       self.L = np.zeros((self.m + self.n + 1, self.n + 1), dtype=int)
       
       # First column
       self.L[:self.m, 0] = -N * self.d
       self.L[self.m:self.m + self.n, 0] = -self.rmax
       self.L[self.m + self.n, 0] = self.rmax
       
       # Top-right block
       self.L[:self.m, 1:] = N * self.A
       
       # Middle diagonal block
       for i in range(self.n):
           self.L[self.m + i, 1 + i] = 2 * self.c[i]
   
   def _get_reduced_basis(self):
       ext_m, ext_n = self.L.shape
       L_lll = IntegerMatrix.from_matrix(self.L.T.tolist())
       LLL.reduction(L_lll)
       L_reduced = np.array(
           [[L_lll[i][j] for j in range(ext_m)] for i in range(ext_n)], dtype=int
       ).T
       
       b = []
       for j in range(self.n - self.m + 1):
           col = L_reduced[self.m:, j]
           b.append(col)
       self.basis = np.array(b).T
   
   def _get_gso(self):
       basis_as_rows = self.basis.T
       n_vectors = basis_as_rows.shape[0]
       
       A = IntegerMatrix.from_matrix(basis_as_rows.tolist())
       M = GSO.Mat(A)
       M.update_gso()
       
       # Extract mu coefficients
       self.mu = np.zeros((n_vectors, n_vectors))
       for i in range(n_vectors):
           self.mu[i, i] = 1.0
           for j in range(i):
               self.mu[i, j] = M.get_mu(i, j)
       
       # Extract squared norms
       self.b_hat_norms_sq = np.zeros(n_vectors)
       for i in range(n_vectors):
           self.b_hat_norms_sq[i] = M.get_r(i, i)
       
       # Compute orthogonal vectors
       self.b_hat = []
       for i in range(n_vectors):
           b_hat_i = self.basis[:, i].astype(float)
           for j in range(i):
               b_hat_i = b_hat_i - self.mu[i, j] * self.b_hat[j]
           self.b_hat.append(b_hat_i)
   
   def _compute_dual_norms(self):
       # Compute dual basis norms for pruning strategy 2
       n_vectors = self.n - self.m + 1
       
       # Dual basis: b_bar^(i) · b^(j) = δ_ij
       # b_bar = (B^T B)^(-1) B^T where B is the basis matrix
       B = self.basis
       gram = B.T @ B
       gram_inv = np.linalg.inv(gram)
       self.b_bar = B @ gram_inv
       
       # Store L2 and L1 norms of dual basis vectors
       self.b_bar_norms = {
           'l2': np.array([np.linalg.norm(self.b_bar[:, i], 2) for i in range(n_vectors)]),
           'l1': np.array([np.linalg.norm(self.b_bar[:, i], 1) for i in range(n_vectors)])
       }
   
   # Pruning strategy 1: Norm bound
   def prune_norm(self, w_norm_sq):
       c = (self.n + 1) * self.rmax ** 2
       return w_norm_sq > c
   
   # Pruning strategy 2: Cauchy-Schwarz bound
   def get_u_bounds(self):
       n_vectors = self.n - self.m + 1
       c = np.sqrt((self.n + 1) * self.rmax ** 2)
       
       u_bounds = {
           'l2': np.array([self.b_bar_norms['l2'][i] * c for i in range(n_vectors)]),
           'l1': np.array([self.b_bar_norms['l1'][i] * self.rmax for i in range(n_vectors)])
       }
       return u_bounds
   
   # Pruning strategy 3: Hölder's inequality test
   def prune_holder(self, w):
       w_norm_sq = np.dot(w, w)
       w_norm_l1 = np.sum(np.abs(w))
       return w_norm_sq > self.rmax * w_norm_l1


def test_dual_basis(ms):
   """Test if dual basis is computed correctly"""
   n_vectors = ms.n - ms.m + 1
   
   # Check b_bar^(i) · b^(j) = δ_ij
   print("Testing dual basis property: b_bar^(i) · b^(j) = δ_ij")
   print("-" * 50)
   
   max_error = 0
   for i in range(n_vectors):
       for j in range(n_vectors):
           dot_product = np.dot(ms.b_bar[:, i], ms.basis[:, j])
           expected = 1.0 if i == j else 0.0
           error = abs(dot_product - expected)
           max_error = max(max_error, error)
           
           if i < 3 and j < 3:  # Print first few for inspection
               print(f"b_bar[{i}] · b[{j}] = {dot_product:.6f}, expected = {expected}")
   
   print(f"\nMax error in dual basis property: {max_error:.2e}")
   
   # Verify u = b_bar^T @ v for any v = B @ u
   print("\nTesting coordinate extraction: u = b_bar^T @ v")
   print("-" * 50)
   
   # Random test vector
   u_test = np.random.randn(n_vectors)
   v_test = ms.basis @ u_test
   u_recovered = ms.b_bar.T @ v_test
   
   recovery_error = np.linalg.norm(u_test - u_recovered)
   print(f"Original u: {u_test[:3]}...")
   print(f"Recovered u: {u_recovered[:3]}...")
   print(f"Recovery error: {recovery_error:.2e}")
   
   return max_error < 1e-10 and recovery_error < 1e-10


# Test example
if __name__ == "__main__":
   m, n = 3, 20
   A = np.array([
       [46, 8, 29, 40, 28, 11, 34, 31, 31, 50, 13, 40, 27, 44, 25, 27, 16, 43, 45, 10],
       [40, 15, 23, 38, 15, 36, 46, 1, 26, 27, 23, 1, 6, 45, 7, 42, 10, 17, 10, 7],
       [16, 6, 10, 31, 0, 32, 22, 43, 47, 38, 30, 45, 5, 23, 12, 22, 26, 47, 15, 45]
   ], dtype=int)
   d = np.array([299, 217, 257], dtype=int)
   
   ms = MarketSplit(A, d)
   
   if test_dual_basis(ms):
       print("\n✓ Dual basis test PASSED")
   else:
       print("\n✗ Dual basis test FAILED")
