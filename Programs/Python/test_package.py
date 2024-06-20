from numba import cuda

@cuda.jit(device=True)
def test_fun():
  print('test ok')