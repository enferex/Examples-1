/*
* @@name:	device.1c
* @@type:	C
* @@compilable:	yes
* @@linkable:	no
* @@expect:	success
*/
#include <stdio.h>
#include <omp.h>
#pragma omp declare target
void vec_mult(float *p, float *v1, float *v2, int N);
extern float *p, *v1, *v2;
extern int N;
#pragma omp end declare target
extern void init_vars(float *, float *, int);
extern void output(float *, int);
void native_vecmult(float *p, float *v1, float *v2, int iters)
{
    // Arguments are now native device pointers, not host pointers
    if(HAVE_CUDA) {
      cuda_vecmult(p, v1, v2, N);
    } else if (HAVE_PHI) {
      phi_vecmult(p, v1, v2, N);
    } else {
      #pragma omp target teams distribute parallel for \
        private(i) num_threads(nthreads) is_device(p, v1, v2)
      for (int i=0; i<iters; i++)
        p[i] = v1[i] * v2[i];
    }
}
void foo()
{
  init_vars(p, v1, v2, N);
  // Map data into the device with OpenMP
  #pragma omp target data map(p[:N], v1[:N], v2[:N])
  {
    // Get native device pointers
    #pragma omp target data use_device(p, v1, v2)
    native_vecmult(p, v1, v2);
  }
  output(p, N);
}
