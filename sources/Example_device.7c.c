/*
* @@name:	device.1c
* @@type:	C
* @@compilable:	yes
* @@linkable:	no
* @@expect:	success
*/
#include <stdio.h>
#include <omp.h>
void update_halo(float *dst, float *src, 
                 int width, int height, int halo_width,
                 int from, int to)
{
    // Top edge
    omp_target_memcpy_rect(
            dst, src,
            sizeof(float),
            2,
            (size_t[]){halo_width, width},//volume
            (size_t[]){0, 0},//dst_offset
            (size_t[]){0, 0},//src_offset
            (size_t[]){height, width},// dst_dimensions
            (size_t[]){height, width},// src_dimensions
            to, from);
    // Bottom edge
    omp_target_memcpy_rect(
            dst, src,
            sizeof(float),
            2,
            (size_t[]){halo_width, width},//volume
            (size_t[]){height - 1 - halo_width, 0},//dst_offset
            (size_t[]){height - 1 - halo_width, 0},//src_offset
            (size_t[]){height, width},// dst_dimensions
            (size_t[]){height, width},// src_dimensions
            to, from);
    // Left edge
    omp_target_memcpy_rect(
            dst, src,
            sizeof(float),
            2,
            (size_t[]){height, halo_width},//volume
            (size_t[]){0, 0},//dst_offset
            (size_t[]){0, 0},//src_offset
            (size_t[]){height, width},// dst_dimensions
            (size_t[]){height, width},// src_dimensions
            to, from);
    // Right edge
    omp_target_memcpy_rect(
            dst, src,
            sizeof(float),
            2,
            (size_t[]){height, halo_width},//volume
            (size_t[]){0, width - 1 - halo_width},//dst_offset
            (size_t[]){0, width - 1 - halo_width},//src_offset
            (size_t[]){height, width},// dst_dimensions
            (size_t[]){height, width},// src_dimensions
            to, from);
}
void update_center(float *dst, float *src, 
                   int width, int height, int halo_width,
                   int from, int to)
{
    // All but the halo
    omp_target_memcpy_rect(
            dst, src,
            sizeof(float),
            2,
            (size_t[]){height - halo_width * 2, width - halo_width * 2},//volume
            (size_t[]){halo_width, halo_width},//dst_offset
            (size_t[]){halo_width, halo_width},//src_offset
            (size_t[]){height, width},// dst_dimensions
            (size_t[]){height, width},// src_dimensions
            to, from);
}
