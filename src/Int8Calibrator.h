#ifndef _ENTROY_CALIBRATOR_H
#define _ENTROY_CALIBRATOR_H

#include <cudnn.h>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "utils.h"

nvinfer1::IInt8Calibrator* GetInt8Calibrator(const std::string& calibratorType,
                int batchSize,const std::string& dataPath,
                const std::string& calibrateCachePath);

class TrtInt8Calibrator : public nvinfer1::IInt8Calibrator {
public:
    TrtInt8Calibrator(const std::string& calibratorType, 
        const int batchSize, const std::string& dataPath,
        const std::string& calibrateCachePath);

    virtual ~TrtInt8Calibrator();

    int getBatchSize() const noexcept override;

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    const void* readCalibrationCache(size_t& length) noexcept override;

    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

    nvinfer1::CalibrationAlgoType getAlgorithm() noexcept override;

private:
    std::string mCalibratorType;
    int mBatchSize;
    std::vector<std::string> mFileList;
    std::string mCalibrateCachePath;
    int mCurBatchIdx=0;
    int mCount;
    std::vector<void*> mDeviceBatchData;
    std::vector<char> mCalibrationCache;
};

#endif //_ENTROY_CALIBRATOR_H
