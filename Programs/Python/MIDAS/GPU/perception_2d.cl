inline float phase(float x, float y){
  if (x > 0)                { return atan(y/x); }
  else if (x < 0 && y >= 0) { return atan(y/x) + M_PI_F; }
  else if (x < 0 && y < 0)  { return atan(y/x) - M_PI_F; }
  else if (x == 0 && y > 0) { return M_PI_F/2; }
  else if (x == 0 && y < 0) { return -M_PI_F/2; }
  else                      { return 0; }
}

inline float angle(float a){
  while (a>=2*M_PI_F) { a -= 2*M_PI_F; }
  while (a<0) { a += 2*M_PI_F; }
  return a;
}

__kernel void perception(
  __global const float *arena,
  __global const bool  *bcond,
  __global const int *pairs,
  __global float *X,
  __global float *Y,
  __global float *A,
  __global const float *_r,         // Canva
  __global const float *_theta,
  __global uint *In
    ) {

  // ─── Definitions ────────────────────────────

  // Pair id
  uint id = get_global_id(0);

  // Agents id
  uint i1 = pairs[2*id];
  uint i2 = pairs[2*id+1];

  // Positions and orientations
  float x1 = X[i1];
  float y1 = Y[i1];
  float a1 = A[i1];

  float x2 = X[i2];
  float y2 = Y[i2];
  float a2 = A[i2];

  // Canva size
  uint n_r = _r[0];
  uint n_theta = _theta[0];

  // ─── Interaction ────────────────────────────

  // ─── Periodic boundary conditions

  if (arena[0]==1.0) {

    // X-axis
    if (bcond[0] && fabs(x2-x1)>arena[1]) {
      if (x1<x2) { x2 -= 2*arena[1]; }
      else { x2 += 2*arena[1]; }
    }

    // Y-axis
    if (bcond[1] && fabs(y2-y1)>arena[2]) {
      if (y1<y2) { y2 -= 2*arena[2]; }
      else { y2 += 2*arena[2]; }
    }

  }

  // ─── Check

  float r2 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
  if (r2 > _r[n_r]*_r[n_r]) { return; }

  // ─── Distance

  // NB: The same distance is shared by both agents

  uint ir = n_r - 1;
  for (uint k=0; k<n_r-1; k++) {
    if (r2 <= _r[k+1]*_r[k+1]) { 
      ir = k; 
      break; }
  }

  // ─── Angles

  float phi = phase(x2-x1, y2-y1);
  a1 = angle(phi - a1);
  a2 = angle(M_PI_F + phi - a2);

  // Slice angle
  float da = 2*M_PI_F/n_theta;
  
  // First angular index
  uint ia1 = n_theta - 1;
  for (uint k=0; k<n_theta-1; k++) {
    if (a1 <= (k+1)*da) { 
      ia1 = k; 
      break; }
  }

  // Second angular index
  uint ia2 = n_theta - 1;
  for (uint k=0; k<n_theta-1; k++) {
    if (a2 <= (k+1)*da) { 
      ia2 = k; 
      break; }
  }

  // ─── Input increment ────────────────────────

  // Atomic increment to build the density inputs
  atom_inc(&In[i1*n_r*n_theta + ir*n_theta + ia1]);
  atom_inc(&In[i2*n_r*n_theta + ir*n_theta + ia2]);

}

