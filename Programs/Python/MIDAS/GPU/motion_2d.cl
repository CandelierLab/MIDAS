__kernel void motion(
  __global const float *arena,
  __global const bool  *bcond,
  __global float *X,
  __global float *Y,
  __global float *V,
  __global float *A
    ) {

  // ─── Definitions ────────────────────────────

  // Agent id
  uint id = get_global_id(0);

  float x0 = X[id];
  float y0 = Y[id];
  float v = V[id];
  float a = A[id];

  // ─── Base motion ────────────────────────────

  float x1 = x0 + v*cos(a);
  float y1 = y0 + v*sin(a);

  X[id] = x1;
  Y[id] = y1;

  // ─── Arena ──────────────────────────────────

  // ─── Circular arena
  if (arena[0]==0.0) {

    float r2 = x1*x1 + y1*y1;

    // Switch to complex coordinates
    if (r2 > arena[1]*arena[1]) {

      float phi = a + asin((y0*cos(a) - x0*sin(a))/arena[1]);

      float xc = arena[1]*cos(phi);
      float yc = arena[1]*sin(phi);
      float r = v - sqrt((xc-x0)*(xc-x0) + (yc-y0)*(yc-y0));

      // Update orientation and position
      A[id] = M_PI_F + 2*phi - a;
      X[id] = xc + r*cos(A[id]);
      Y[id] = yc + r*sin(A[id]);
      
    }
  }
  
  // ─── Rectangular arena  
  if (arena[0]==1.0) {

    // Low X
    if (x1 < -arena[1]) {
      if (bcond[0]) { 
        X[id] = x1 + 2*arena[1];        // Periodic
      } else { 
        X[id] = -x1 - 2*arena[1];       // Bouncing
        A[id] = M_PI_F - a;
      }
    }

    // High X
    if (x1 > arena[1]) {
      if (bcond[0]) { 
        X[id] = x1 - 2*arena[1];        // Periodic
      } else { 
        X[id] = -x1 + 2*arena[1];       // Bouncing
        A[id] = M_PI_F - a;
      }
    }

    // Low Y
    if (y1 < -arena[2]) {
      if (bcond[1]) { 
        Y[id] = y1 + 2*arena[2];        // Periodic
      } else { 
        Y[id] = -y1 - 2*arena[2];       // Bouncing
        A[id] = -a;
      }
    }

    // High Y
    if (y1 > arena[2]) {
      if (bcond[1]) { 
        Y[id] = y1 - 2*arena[2];        // Periodic
      } else { 
        Y[id] = -y1 + 2*arena[2];       // Bouncing
        A[id] = -a;
      }
    }

  }

}