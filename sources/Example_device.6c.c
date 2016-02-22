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
extern float *p, *v1, *v2;
extern int N;// must be a multiple of 100
#pragma omp end declare target
extern void init_vars(float *, float *, int);
extern void output(float *, int);
void omp_vec_mult(float *p, float *v1, float *v2, int N)
{
    #pragma omp target teams distribute parallel for \
      private(i) num_threads(nthreads) \
      map(always, to: v1[:N], v2[:N]) \
      map(always, from: p[:N])
    for (int i=0; i<segment_size; i++)
      p[i] = v1[i] * v2[i];
}
void foo()
{
  init_vars(p, v1, v2, N);
  int dev = omp_get_default_device();
  int segment_size = 100;

  float *dp  = omp_target_alloc(segment_size * sizeof(float), dev);
  float *dv1 = omp_target_alloc(segment_size * sizeof(float), dev);
  float *dv2 = omp_target_alloc(segment_size * sizeof(float), dev);

  for (int tile_offset = 0; tile_offset < N; tile_offset += segment_size){
    // No new allocation of p, or v, and no explicit copies here, map will
    // take care of that
    omp_target_associate_ptr(p + tile_offset, dp, segment_size * sizeof(float), 0, dev);
    omp_target_associate_ptr(v1 + tile_offset, dv1, segment_size * sizeof(float), 0, dev);
    omp_target_associate_ptr(v2 + tile_offset, dv2, segment_size * sizeof(float), 0, dev);
    omp_vec_mult(p, v1, v2, N);
    omp_target_disassociate_ptr(dp, dev);
    omp_target_disassociate_ptr(dv1, dev);
    omp_target_disassociate_ptr(dv2, dev);
  }

  omp_target_free (dp, dev);
  omp_target_free (dv1, dev);
  omp_target_free (dv2, dev);

  output(p, N);
}
