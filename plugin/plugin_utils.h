/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2019-12-04 14:26:15
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-12-11 14:50:31
 */
#ifndef PLUGIN_UTILS_H
#define PLUGIN_UTILS_H

#include "NvInfer.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <iostream>

#include "utils.h"

/**
 * @description: these are some common function during write your custom plugin,
 *               you can include this header and use it directly.
 * @描述: 这是实现自定义插件的时候非常常用的一些函数,我将它们抽离出来,你可以直接调用即可
 */

// this is for debug, and you can find a lot assert in plugin implementation,
// it will reduce the time you spend on debug
// 为了方便debug, 你也可以在插件的实现里面看到我大量使用了assert, 这个可以大大减少花在
// debug上的时间
#define ASSERT(assertion)                                                                                              \
{                                                                                                                  \
    if (!(assertion))                                                                                              \
    {                                                                                                              \
        std::cerr << "#assertion fail " << __FILE__ << " line " << __LINE__ << std::endl;                                     \
        abort();                                                                                                   \
    }                                                                                                              \
}

// write value to buffer
template <typename T>
void write(char *&buffer, const T &val)
{
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

// read value from buffer
template <typename T>
void read(const char *&buffer, T &val)
{
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
}

// return needed space of a datatype
size_t type2size(nvinfer1::DataType type);

// copy data to device memory
void* copyToDevice(const void* data, size_t count);

// copy data to buffer.
void copyToBuffer(char*& buffer, const void* data, size_t count);

// convert data to datatype and copy it to device
void convertAndCopyToDeivce(void*& deviceWeights, const nvinfer1::Weights &weights,
                            nvinfer1::DataType datatype);

// convert data to datatype and copy it to buffer
void convertAndCopyToBuffer(char*& buffer, const nvinfer1::Weights weights,
                            nvinfer1::DataType datatype);

// deserialize buffer to device memory.
void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size);

#endif //PLUGIN_UTILS_H