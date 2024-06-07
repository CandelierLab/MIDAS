from numba import cuda

@cuda.jit(device=True)
def test():
  print('ok')
