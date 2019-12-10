/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2019-12-04 14:26:15
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-12-10 16:24:45
 */
#ifndef PLUGIN_UTILS_H
#define PLUGIN_UTILS_H

#include "NvInfer.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <iostream>

#include "utils.h"

static const char* G_PLUGIN_NAMESPACE = "_TRT";
static const char* G_PLUGIN_VERSION = "1";

#define ASSERT(assertion)                                                                                              \
{                                                                                                                  \
    if (!(assertion))                                                                                              \
    {                                                                                                              \
        std::cerr << "#assertion fail " << __FILE__ << " line " << __LINE__ << std::endl;                                     \
        abort();                                                                                                   \
    }                                                                                                              \
}

template <typename T>
void write(char *&buffer, const T &val)
{
    *reinterpret_cast<T *>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void read(const char *&buffer, T &val)
{
    val = *reinterpret_cast<const T *>(buffer);
    buffer += sizeof(T);
}

size_t type2size(nvinfer1::DataType type);

void* copyToDevice(const void* data, size_t count);

void convertAndCopyToDeivce(void*& deviceWeights, const nvinfer1::Weights &weights,
                            nvinfer1::DataType datatype);

void convertAndCopyToBuffer(char*& buffer, const nvinfer1::Weights weights,
                            nvinfer1::DataType datatype);

void deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size);

#endif //PLUGIN_UTILS_H