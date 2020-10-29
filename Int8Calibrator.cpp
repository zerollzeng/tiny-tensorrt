/*
 * @Description: int8 entrophy calibrator 2
 * @Author: zengren
 * @Date: 2019-08-21 16:52:06
 * @LastEditTime: 2019-08-22 17:04:49
 * @LastEditors: Please set LastEditors
 */
#include "Int8Calibrator.h"
#include <fstream>
#include <iterator>
#include <cassert>
#include <string.h>
#include <algorithm>

nvinfer1::IInt8Calibrator* GetInt8Calibrator(const std::string& calibratorType, 
											 int BatchSize,
											 const std::vector<std::vector<float>>& data,
											 const std::string& CalibDataName,
											 bool readCache) {
    if(calibratorType == "Int8EntropyCalibrator2") {
        return new Int8EntropyCalibrator2(BatchSize,data,CalibDataName,readCache);
    } else {
        std::cout << "unsupport calibrator type" << std::endl;
        assert(false);
    }
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int BatchSize,const std::vector<std::vector<float>>& data,
                                        const std::string& CalibDataName /*= ""*/,bool readCache /*= true*/)
    : mCalibDataName(CalibDataName),mBatchSize(BatchSize),mReadCache(readCache)
{     
    mDatas.reserve(data.size());
    mDatas = data;

    mInputCount =  BatchSize * data[0].size();
    mCurBatchData = new float[mInputCount];
    mCurBatchIdx = 0;
    CUDA_CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
}


Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    CUDA_CHECK(cudaFree(mDeviceInput));
    if(mCurBatchData)
        delete[] mCurBatchData;
}


bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    std::cout << "name: " << names[0] << "nbBindings: " << nbBindings << std::endl;
    if (mCurBatchIdx + mBatchSize > int(mDatas.size())) 
            return false;

    float* ptr = mCurBatchData;
    size_t imgSize = mInputCount / mBatchSize;
    auto iter = mDatas.begin() + mCurBatchIdx;

    std::for_each(iter, iter + mBatchSize, [=,&ptr](std::vector<float>& val){
        assert(imgSize == val.size());
        memcpy(ptr,val.data(),imgSize*sizeof(float));
        
        ptr += imgSize;
    });

    CUDA_CHECK(cudaMemcpy(mDeviceInput, mCurBatchData, mInputCount * sizeof(float), cudaMemcpyHostToDevice));
    //std::cout << "input name " << names[0] << std::endl;
    bindings[0] = mDeviceInput;

    std::cout << "load batch " << mCurBatchIdx << " to " << mCurBatchIdx + mBatchSize - 1 << std::endl;        
    mCurBatchIdx += mBatchSize;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length)
{
    mCalibrationCache.clear();
    std::ifstream input(mCalibDataName+".calib", std::ios::binary);
    input >> std::noskipws;
    if (mReadCache && input.good())
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length)
{
    std::ofstream output(mCalibDataName+".calib", std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

