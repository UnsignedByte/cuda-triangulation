#include <cstdio>
#include <cassert>
#include "helper_cuda.cuh"
#include "evaluate.cuh"
#include <chrono>
#include <algorithm>
#include <cstdint>
#define TESTS 100000
#define BLOCK_SIZE 16
#define N_STREAMS 100
// *image is an image with width full_width.
// width and height correspond to the size of the sub-image to draw in
// offset is the position of the sub-image in the full image
__global__ void draw(Triangle tri, bool *image, int full_width, int2 topleft, int2 bottomright)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x + topleft.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y + topleft.y;

  if (x > bottomright.x || y > bottomright.y)
  {
    return;
  }

  int idx = y * full_width + x;

  image[idx] = in_triangle(tri, make_int2(x, y));
}

// generate a random CCW triangle
Triangle random_triangle(int2 dims)
{
  Triangle tri;
  tri.a = make_int2(rand() % dims.x, rand() % dims.y);
  tri.b = make_int2(rand() % dims.x, rand() % dims.y);
  tri.c = make_int2(rand() % dims.x, rand() % dims.y);

  // ensure CCW
  int area = -tri.b.y * tri.c.x + tri.a.y * (tri.c.x - tri.b.x) + tri.a.x * (tri.b.y - tri.c.y) + tri.b.x * tri.c.y;

  if (area < 0)
  {
    int2 tmp = tri.a;
    tri.a = tri.b;
    tri.b = tmp;
  }

  return tri;
}

// dispatch the drawing of the triangle
template <int blockSize>
uint32_t draw_triangle(cudaStream_t stream, Triangle tri, bool *image, int full_width)
{
  // figure out the sub-image that this triangle fits in
  int2 topleft;
  topleft.x = std::min(std::min(tri.a.x, tri.b.x), tri.c.x);
  topleft.y = std::min(std::min(tri.a.y, tri.b.y), tri.c.y);

  int2 bottomright;
  bottomright.x = std::max(std::max(tri.a.x, tri.b.x), tri.c.x);
  bottomright.y = std::max(std::max(tri.a.y, tri.b.y), tri.c.y);

  dim3 block(blockSize, blockSize);
  dim3 grid((bottomright.x - topleft.x + blockSize - 1) / blockSize, (bottomright.y - topleft.y + blockSize - 1) / blockSize);

  draw<<<grid, block, 0, stream>>>(tri, image, full_width, topleft, bottomright);

  return (bottomright.x - topleft.x) * (bottomright.y - topleft.y);
}

int main()
{
  statDevice();

  // Size of the image
  int2 dims = make_int2(1920, 1080);

  bool *images[N_STREAMS];
  uint32_t work_assigned[N_STREAMS] = {0};
  cudaStream_t streams[N_STREAMS];

  size_t mem_size = dims.x * dims.y * sizeof(bool);

  // allocate memory for the images
  for (int i = 0; i < N_STREAMS; ++i)
  {
    checkCudaErrors(cudaMalloc(&images[i], mem_size));
    checkCudaErrors(cudaMemset(images[i], 0, mem_size));

    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  // Start the timer
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < TESTS; ++i)
  {
    Triangle tri = random_triangle(dims);

    // Assign the work to the stream with the least amount of work
    int min_idx = 0;
    for (int j = 1; j < N_STREAMS; ++j)
    {
      if (work_assigned[j] < work_assigned[min_idx])
      {
        min_idx = j;
      }
    }

    work_assigned[min_idx] += draw_triangle<BLOCK_SIZE>(streams[min_idx], tri, images[min_idx], dims.x);

    // Now clear the image
    checkCudaErrors(cudaMemsetAsync(images[min_idx], 0, mem_size, streams[min_idx]));
  }

  // Wait for all streams to finish
  for (int i = 0; i < N_STREAMS; ++i)
  {
    checkCudaErrors(cudaStreamSynchronize(streams[i]));
  }

  auto end = std::chrono::high_resolution_clock::now();

  // Print the time
  printf("Average time (ms): %lf\n", (double)std::chrono::duration_cast<std::chrono::nanoseconds>((end - start) / TESTS).count() / 1000000.);

  // Free the memory
  for (int i = 0; i < N_STREAMS; ++i)
  {
    checkCudaErrors(cudaFree(images[i]));
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  }
}