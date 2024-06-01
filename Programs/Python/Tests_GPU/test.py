import os

import numpy as np
from numba import jit
from numba import cuda

os.system('clear')

@cuda.jit
def distance_matrix(mat, out):

  i, j = cuda.grid(2)
  if i<mat.shape[0] and j<=i:
      
    # d = 0
    # for k in range(n):
    #   tmp = mat[i, k] - mat[j, k]
    #   d += tmp * tmp

    out[i, j] = mat[i,0] + mat[j,0]


n = 500
h_A = np.stack((np.arange(n), np.zeros(n))).T

d_A = cuda.to_device(h_A)
d_D = cuda.device_array((n, n))

block_dim = (16, 16)
grid_dim = (int(n/block_dim[0] + 1), int(n/block_dim[1] + 1))

distance_matrix[grid_dim, block_dim](d_A, d_D)
h_D = d_D.copy_to_host()

print(h_D)
