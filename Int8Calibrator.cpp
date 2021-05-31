/*
 * @Description: int8 entrophy calibrator 2
 * @Author: zengren
 * @Date: 2019-08-21 16:52:06
 * @LastEditTime: 2019-08-22 17:04:49
 * @LastEditors: Please set LastEditors
 */
#include "Int8Calibrator.h"
#include "spdlog/spdlog.h"
#include "cnpy.h"
#include "utils.h"

#include <fstream>
#include <iterator>
#include <cassert>
#include <string.h>
#include <algorithm>

#include <sys/types.h>
#include <dirent.h>

void read_directory(const std::string& name, std::vector<std::string>& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        if(strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0 ) {
            continue;
        }
        v.push_back(name+dp->d_name);
    }
    closedir(dirp);
}

nvinfer1::IInt8Calibrator* GetInt8Calibrator(const std::string& calibratorType,
                int batchSize,const std::string& dataPath, 
                const std::string& calibrateCachePath) {
    if(calibratorType == "Int8EntropyCalibrator2") {
        return new Int8EntropyCalibrator2(batchSize,dataPath,calibrateCachePath);
    } else {
        spdlog::error("unsupported calibrator type");
        assert(false);
    }
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


Int8EntropyCalibrator2::Int8EntropyCalibrator2(const int batchSize, const std::string& dataPath, 
                                               const std::string& calibrateCachePath)
{
    spdlog::info("init calibrator...");
    mBatchSize = batchSize;
    mCalibrateCachePath = calibrateCachePath;

    if(dataPath != "") {
        std::string path = dataPath;
        if(!ends_with(path, "/")) {
            path = path + "/";
        }
        read_directory(path, mFileList);
        mCount = mFileList.size();
        assert(mCount != 0);

        cnpy::npz_t data = cnpy::npz_load(mFileList[0]);
        mDeviceBatchData.resize(data.size());
        int i=0;
        for(cnpy::npz_t::iterator it = data.begin(); it != data.end(); ++it) {
            cnpy::NpyArray np_array = it->second;
            size_t num_bytes = np_array.num_bytes();
            mDeviceBatchData[i] = safeCudaMalloc(num_bytes * mBatchSize);
            i++;
        }
    }    
}


Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    for(size_t i=0;i<mDeviceBatchData.size();i++) {
        safeCudaFree(mDeviceBatchData[i]);
    }
}

int Int8EntropyCalibrator2::getBatchSize() const{
    spdlog::info("get batch size {}", mBatchSize);
    return mBatchSize;
}


bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    if (mCurBatchIdx + mBatchSize > mCount) {
        return false;
    }

    for(int i=0;i<mBatchSize; i++) {
        cnpy::npz_t data = cnpy::npz_load(mFileList[mCurBatchIdx]);
        for(int j=0;j<nbBindings;j++) {
            const char* name = names[j];
            cnpy::NpyArray np_array = data[name];
            size_t num_bytes = np_array.num_bytes();
            float* arr = np_array.data<float>();
            void* p = static_cast<char*>(mDeviceBatchData[j]) + i*num_bytes;
            CUDA_CHECK(cudaMemcpy(p, arr, num_bytes, cudaMemcpyHostToDevice));
        }
        mCurBatchIdx++;
    }

    for(int j=0;j<nbBindings;j++) {
        bindings[j] = mDeviceBatchData[j];
    }
    spdlog::info("load catlibrate data {}/{} done", mCurBatchIdx, mCount);
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length)
{
    spdlog::info("read calibration cache");
    mCalibrationCache.clear();
    std::ifstream input(mCalibrateCachePath, std::ios::binary);
    input >> std::noskipws;
    if (input.good()) {
        std::copy(std::istream_iterator<char>(input),
                  std::istream_iterator<char>(), 
                  std::back_inserter(mCalibrationCache));
    }

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length)
{
    spdlog::info("write calibration cache");
    std::ofstream output(mCalibrateCachePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

