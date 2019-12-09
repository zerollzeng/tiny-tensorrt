/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2019-12-04 14:26:15
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-12-09 11:12:30
 */
#ifndef PLUGIN_COMMOM_H
#define PLUGIN_COMMOM_H

#include "NvInfer.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <iostream>

#include "utils.h"

#define ASSERT(assertion)                                                                                              \
{                                                                                                                  \
    if (!(assertion))                                                                                              \
    {                                                                                                              \
        std::cerr << "#assertion" << __FILE__ << "," << __LINE__ << std::endl;                                     \
        abort();                                                                                                   \
    }                                                                                                              \
}

static const char* G_PLUGIN_NAMESPACE = "_TRT";
static const char* G_PLUGIN_VERSION = "1";

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

#endif //PLUGIN_COMMOM_H