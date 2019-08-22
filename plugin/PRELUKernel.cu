#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <plugin/PRELUKernel.cuh>

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

/******** PReLU CUDA function ********/
// CUDA kernele for forward
__global__ void PReLUForward(const int n, const int channels, const int dim,
    const float* slope_data,
    const float* in, float* out,
    const float zero,
    const int div_factor) {
    CUDA_KERNEL_LOOP(index, n) {
        int c = (index / dim) % channels / div_factor;
        out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
    }
}


cudaError_t Forward_gpu(const int count, const int channels, const int dim,
                const float* mDeviceKernel,
                const float* bottom_data, float* top_data, 
                const float zero,
                const int div_factor, const cudaStream_t stream) {
    PReLUForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream>>>
        (count, channels, dim, mDeviceKernel, bottom_data, top_data, zero, div_factor);
    cudaError_t err = cudaGetLastError();
    return err;
}