/*
 * @Description: Plugin Factory
 * @Author: zengren
 * @Date: 2019-07-16 11:17:12
 * @LastEditTime: 2019-08-15 14:23:07
 * @LastEditors: Please set LastEditors
 */


#ifndef PLUGIN_FACTORY_HPP
#define PLUGIN_FACTORY_HPP

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"

#include <map>

using namespace nvinfer1;


// integration for serialization
class PluginFactory : public nvcaffeparser1::IPluginFactoryV2 {
public:
    PluginFactory(int yoloClassNum);

    virtual ~PluginFactory() {}
    // ------------------inherit from IPluginFactoryV2--------------------
    // determines if a layer configuration is provided by an IPluginV2
    virtual bool isPluginV2(const char* layerName) override;

    // create a plugin
    virtual IPluginV2* createPlugin(const char* layerName, const Weights* weights, int nbWeights, const char* libNamespace="") override;

private:
    int mYoloClassNum;

};


#endif