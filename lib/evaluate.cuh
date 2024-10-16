#ifndef EVALUATE_H
#define EVALUATE_H

#include <stdio.h>
#include "helper_cuda.cuh"
#define FULL_MASK 0xffffffff

typedef struct
{
  int2 a, b, c;
} Triangle;

/// https://stackoverflow.com/a/14382692/5844752
/// Assumes that the triangle is CCW
__device__ bool in_triangle(const Triangle &tri, const int2 &p)
{
  int s = tri.a.y * tri.c.x - tri.a.x * tri.c.y + (tri.c.y - tri.a.y) * p.x + (tri.a.x - tri.c.x) * p.y;
  int t = tri.a.x * tri.b.y - tri.a.y * tri.b.x + (tri.a.y - tri.b.y) * p.x + (tri.b.x - tri.a.x) * p.y;

  int area_double = -tri.b.y * tri.c.x + tri.a.y * (tri.c.x - tri.b.x) + tri.a.x * (tri.b.y - tri.c.y) + tri.b.x * tri.c.y;

  return s > 0 && t > 0 && (s + t) < area_double;
}

// The data that needs to be stored to properly calculate MSE.
// Uses the fact that MSE = sum((x_i - m)^2)/n = sum(x_i^2 - 2 * x_i * m + m^2)/n = sum(x_i^2)/n - 2 * m * sum(x_i)/n + m^2 = sum(x_i^2)/n - m^2
typedef struct
{
  int n;
  int3 sum;
  int sum_sq;
} MSEData;

__device__ void sum(MSEData &a, const MSEData &b)
{
  a.n += b.n;
  a.sum.x += b.sum.x;
  a.sum.y += b.sum.y;
  a.sum.z += b.sum.z;
  a.sum_sq += b.sum_sq;
}

template <unsigned int distance>
__device__ void sum_warp(MSEData *sdata, const unsigned int tid)
{
  sdata[tid].n += __shfl_down_sync(FULL_MASK, sdata[tid].n, distance);
  sdata[tid].sum.x += __shfl_down_sync(FULL_MASK, sdata[tid].sum.x, distance);
  sdata[tid].sum.y += __shfl_down_sync(FULL_MASK, sdata[tid].sum.y, distance);
  sdata[tid].sum.z += __shfl_down_sync(FULL_MASK, sdata[tid].sum.z, distance);
  sdata[tid].sum_sq += __shfl_down_sync(FULL_MASK, sdata[tid].sum_sq, distance);
}

template <unsigned int blockSize>
__device__ void mse_warp(MSEData *sdata, const unsigned int tid)
{
  // No need for synchronization here, as we fit into a single warp
  if (blockSize >= 64)
  {
    sum(sdata[tid], sdata[tid + 32]);
  }
  if (blockSize >= 32)
    sum_warp<16>(sdata, tid);
  if (blockSize >= 16)
    sum_warp<8>(sdata, tid);
  if (blockSize >= 8)
    sum_warp<4>(sdata, tid);
  if (blockSize >= 4)
    sum_warp<2>(sdata, tid);
  if (blockSize >= 2)
    sum_warp<1>(sdata, tid);
}

template <unsigned int blockSize>
__global__ void mse(const Triangle &tri, const char3 *image, int full_width, int2 topleft, int2 dims, MSEData *result)
{
  __shared__ MSEData sdata[blockSize];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  sdata[tid].n = 0;
  sdata[tid].sum = make_int3(0, 0, 0);
  sdata[tid].sum_sq = 0;

  if (idx < dims.x * dims.y)
  {
    int2 p;
    p.x = idx % dims.x + topleft.x;
    p.y = idx / dims.x + topleft.y;

    if (in_triangle(tri, p))
    {
      int img_idx = y * full_width + x;

      // Load the pixel value
      char3 pixel = image[img_idx];

      // Update the shared memory
      sdata[tid].n = 1;
      sdata[tid].sum = make_int3(pixel.x, pixel.y, pixel.z);
      sdata[tid].sum_sq = pixel.x * pixel.x + pixel.y * pixel.y + pixel.z * pixel.z;
    }
  }

  __syncthreads();

  // Reduce the shared memory

  if (blockSize >= 512)
  {
    if (tid < 256)
    {
      sum(sdata[tid], sdata[tid + 256]);
    }

    __syncthreads();
  }

  if (blockSize >= 256)
  {
    if (tid < 128)
    {
      sum(sdata[tid], sdata[tid + 128]);
    }

    __syncthreads();
  }

  if (blockSize >= 128)
  {
    if (tid < 64)
    {
      sum(sdata[tid], sdata[tid + 64]);
    }

    __syncthreads();
  }

  if (tid < 32)
  {
    mse_warp<blockSize>(sdata, tid);
  }

  if (tid == 0)
  {
    result[blockIdx.x] = sdata[0];
  }
}

#endif