#ifndef _KERNEL_FUNC_H_
#define _KERNEL_FUNC_H_

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>




// __global__ void PReLUForward(const int n, const int channels, const int dim,
//     const float* slope_data,
//     const float* in, float* out,
//     const float zero,
//     const int div_factor);


// cudaError_t Forward_gpu(const int count, const int channels, const int dim,
//                         const float *mDeviceKernel,
//                         const float *bottom_data, float *top_data,
//                         const float zero,
//                         const int div_factor,
//                         cudaStream_t stream);

template <typename Ftype>
cudaError_t Forward_gpu(const int count, const int channels, const int dim,
                        const Ftype *mDeviceKernel,
                        const Ftype *bottom_data, Ftype *top_data,
                        const Ftype zero,
                        const int div_factor,
                        cudaStream_t stream);


#endif // _KERNEL_FUNC_H_