#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <stdio.h>

// Taken from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
template <typename T>
void __checkCudaErrors(T result, char const *const func, const char *const file,
                       int const line)
{
  if (result)
  {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) __checkCudaErrors((val), #val, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line)
{
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err)
  {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

// Get information about the device
void statDevice()
{
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  printf("Device: %s\n", props.name);
  printf("Warp size: %d\n", props.warpSize);
  printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
  printf("Max blocks per multiprocessor: %d\n", props.maxBlocksPerMultiProcessor);
  printf("Max threads per multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);
  printf("Number of SMs: %d\n", props.multiProcessorCount);
  printf("Memory: %zu\n", props.totalGlobalMem);
}

// Get the optimal grid and block size for a fully parallel kernel
int2 fullyParallelGridDims()
{
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  unsigned int threadsPerBlock = props.maxThreadsPerMultiProcessor / props.maxBlocksPerMultiProcessor;
  unsigned int numBlocks = props.multiProcessorCount * props.maxBlocksPerMultiProcessor;

  return make_int2(numBlocks, threadsPerBlock);
}

#endif // HELPER_CUDA_H