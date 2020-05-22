/*
 * @Description: In User Settings Edit
 * @Author: zerollzeng
 * @Date: 2019-08-23 11:55:03
 * @LastEditTime: 2020-05-22 11:40:34
 * @LastEditors: zerollzeng
 */

#include "PluginFactory.h"
#include "PReLUPlugin/PReLUPlugin.h"
#include "UpSamplePlugin/UpSamplePlugin.hpp"
#include "YoloLayerPlugin/YoloLayerPlugin.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cassert>

PluginFactory::PluginFactory(TrtPluginParams params) {
    spdlog::info("create plugin factory");
    mYoloClassNum = params.yoloClassNum;
    mYolo3NetSize = params.yolo3NetSize;

    mUpsampleScale = params.upsampleScale;
    spdlog::info("yolo3 params: class: {}, netSize: {} ",mYoloClassNum,mYolo3NetSize);
    spdlog::info("upsample params: scale: {}",mUpsampleScale);
}

bool PluginFactory::isPluginV2(const char* layerName) 
{
    std::string strName{layerName};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);
    return (strName.find("prelu") != std::string::npos || strName.find("upsample") != std::string::npos
            || strName.find("yolo-det") != std::string::npos);
}

IPluginV2* PluginFactory::createPlugin(const char *layerName, const Weights* weights, int nbWeights, const char* libNamespace) 
{
    assert(isPluginV2(layerName));

    std::string strName{layerName};
    std::transform(strName.begin(), strName.end(), strName.begin(), ::tolower);

    if (strName.find("prelu") != std::string::npos) {
        // std::cout << "nbWeight: " << nbWeights << std::endl;
        // std::cout << "weights.count: " << weights->count << std::endl;
        return (IPluginV2*)(new PReLUPlugin(weights, nbWeights));
    } 
    else if(strName.find("upsample") != std::string::npos) {
        return (IPluginV2*)(new UpSamplePlugin(mUpsampleScale));
    }
    else if(strName.find("yolo-det") != std::string::npos) {
        return (IPluginV2*)(new YoloLayerPlugin(mYoloClassNum,mYolo3NetSize));
    }
    else
    {
        std::cout << "warning : " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}


