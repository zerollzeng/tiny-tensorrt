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
    return new TrtInt8Calibrator(calibratorType, batchSize,dataPath,calibrateCachePath);
}

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


TrtInt8Calibrator::TrtInt8Calibrator(const std::string& calibratorType, 
    const int batchSize, const std::string& dataPath,
    const std::string& calibrateCachePath)
{
    spdlog::info("init calibrator...");
    mBatchSize = batchSize;
    mCalibrateCachePath = calibrateCachePath;
    mCalibratorType = calibratorType;

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


TrtInt8Calibrator::~TrtInt8Calibrator()
{
    for(size_t i=0;i<mDeviceBatchData.size();i++) {
        safeCudaFree(mDeviceBatchData[i]);
    }
}

int TrtInt8Calibrator::getBatchSize() const noexcept{
    spdlog::info("get batch size {}", mBatchSize);
    return mBatchSize;
}


bool TrtInt8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    spdlog::info("load catlibrate data {}/{}...", mCurBatchIdx, mCount);
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
    return true;
}

const void* TrtInt8Calibrator::readCalibrationCache(size_t& length) noexcept
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

void TrtInt8Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    spdlog::info("write calibration cache");
    std::ofstream output(mCalibrateCachePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

nvinfer1::CalibrationAlgoType TrtInt8Calibrator::getAlgorithm() noexcept
{
    spdlog::info("get calibrator algorithm type");
    if(mCalibratorType == "EntropyCalibratorV2") {
        return nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION_2;
    } else if(mCalibratorType == "EntropyCalibrator") {
        return nvinfer1::CalibrationAlgoType::kENTROPY_CALIBRATION;
    } else if(mCalibratorType == "MinMaxCalibrator") {
        return nvinfer1::CalibrationAlgoType::kMINMAX_CALIBRATION;
    } else {
        assert(false && "unsupported calibrator type");
    }
}
 

