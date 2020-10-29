/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 16:48:34
 * @LastEditTime: 2019-08-22 17:06:20
 * @LastEditors: Please set LastEditors
 */
#ifndef _ENTROY_CALIBRATOR_H
#define _ENTROY_CALIBRATOR_H

#include <cudnn.h>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "utils.h"

nvinfer1::IInt8Calibrator* GetInt8Calibrator(const std::string& calibratorType, 
											 int BatchSize,
											 const std::vector<std::vector<float>>& data,
											 const std::string& CalibDataName,
											 bool readCache);

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
	Int8EntropyCalibrator2(int BatchSize,const std::vector<std::vector<float>>& data,const std::string& CalibDataName = "",bool readCache = true);

	virtual ~Int8EntropyCalibrator2();

	int getBatchSize() const override {
		std::cout << "getbatchSize: " << mBatchSize << std::endl;
		return mBatchSize; 
	}

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

	const void* readCalibrationCache(size_t& length) override;

	void writeCalibrationCache(const void* cache, size_t length) override;

private:
	std::string mCalibDataName;
	std::vector<std::vector<float>> mDatas;
	int mBatchSize;

	int mCurBatchIdx;
	float* mCurBatchData{ nullptr };
	
	size_t mInputCount;
	bool mReadCache;
	void* mDeviceInput{ nullptr };

	std::vector<char> mCalibrationCache;
};

#endif //_ENTROY_CALIBRATOR_H