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
       
       # Run preprocessing
       self._get_extended_matrix()
       self._get_reduced_basis()
       self._get_gso()
   
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
