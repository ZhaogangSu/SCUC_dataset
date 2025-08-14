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

m, n = 3, 20
A = np.array(
[
  [46,  8, 29, 40, 28, 11, 34, 31, 31, 50, 13, 40, 27, 44, 25, 27, 16, 43, 45, 10],
  [40, 15, 23, 38, 15, 36, 46,  1, 26, 27, 23,  1,  6, 45,  7, 42, 10, 17, 10,  7],
  [16,  6, 10, 31,  0, 32, 22, 43, 47, 38, 30, 45,  5, 23, 12, 22, 26, 47, 15, 45]
], dtype=int
)
d = np.array([299, 217, 257], dtype=int)

L, rmax, c = get_extended_matrix(A, d)
print(L)
print(rmax)
print(c)

np.savetxt('ext_mat.txt', L, fmt='%d', delimiter = ' ')

