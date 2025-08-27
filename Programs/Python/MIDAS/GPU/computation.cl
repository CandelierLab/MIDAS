__kernel void computation(
  __global const uint *In,
  __global const float *_r,         // Canva
  __global const float *_theta,
  __global const float *coeff,
  __global float *A
    ) {

  // ─── Definitions ────────────────────────────

  // Agent id
  uint id = get_global_id(0);

  // Canva size
  uint n_r = _r[0];
  uint n_theta = _theta[0];

  float amax = M_PI_F/10;

  // ─── Computation ────────────────────────────

  // Linear combination
  float S = 0;
  for (int i=0; i<n_r; i++) {
    for (int j=0; j<n_theta; j++) {
      S += In[id*n_r*n_theta + i*n_theta + j]*coeff[i*n_theta + j];
    }
  }

  // Activation function
  A[id] += amax*(4/M_PI_F*atan(exp(S/2)) - 1);

}