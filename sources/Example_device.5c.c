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
extern int N;// must be a multiple of 100
#pragma omp end declare target
extern void init_vars(float *, float *, int);
extern void output(float *, int);
void foo()
{
  init_vars(p, v1, v2, N);
  int dev = omp_get_default_device();
  int segment_size = 100;

  float *dv1 = omp_target_alloc(segment_size * sizeof(float), dev);
  float *dv2 = omp_target_alloc(segment_size * sizeof(float), dev);

  for (int tile_offset = 0; tile_offset < N; tile_offset += segment_size){
    omp_target_memcpy(dv1, v1, 
                      segment_size * sizeof(float),
                      0, tile_offset * sizeof(float),
                      dev, omp_get_initial_device());
    omp_target_memcpy(dv2, v2, 
                      segment_size * sizeof(float),
                      0, tile_offset * sizeof(float),
                      dev, omp_get_initial_device());
    #pragma omp target teams distribute parallel for \
      private(i) num_threads(nthreads) \
      is_device(dv1, dv2) \
      map(from: p[tile_offset:segment_size])
    for (int i=0; i<segment_size; i++)
      p[i+tile_offset] = dv1[i] * dv2[i];
  }

  omp_target_free (dv1, dev);
  omp_target_free (dv2, dev);

  output(p, N);
}
