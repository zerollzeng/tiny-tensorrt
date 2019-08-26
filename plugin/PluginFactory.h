/*
 * @Description: Plugin Factory
 * @Author: zengren
 * @Date: 2019-07-16 11:17:12
 * @LastEditTime: 2019-08-23 14:38:54
 * @LastEditors: Please set LastEditors
 */


#ifndef PLUGIN_FACTORY_HPP
#define PLUGIN_FACTORY_HPP

#include "Trt.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"

#include <map>

using namespace nvinfer1;


// integration for serialization
class PluginFactory : public nvcaffeparser1::IPluginFactoryV2 {
public:
    PluginFactory(TrtPluginParams params);

    virtual ~PluginFactory() {}
    // ------------------inherit from IPluginFactoryV2--------------------
    // determines if a layer configuration is provided by an IPluginV2
    virtual bool isPluginV2(const char* layerName) override;

    // create a plugin
    virtual IPluginV2* createPlugin(const char* layerName, const Weights* weights, int nbWeights, const char* libNamespace="") override;

private:
    // yolo-det layer params
    int mYoloClassNum;
    int mYolo3NetSize;

    // upsample layer params
    float mUpsampleScale;
};


#endif