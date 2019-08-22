/* Class Plugin Factory Definitions */
// caffe parser plugin implementation

#include "plugin/PluginFactory.hpp"
#include "plugin/PRELUPlugin.hpp"
#include "plugin/UpSamplePlugin.hpp"
#include "plugin/YoloLayerPlugin.hpp"

#include <algorithm>
#include <cassert>

PluginFactory::PluginFactory(int yoloClassNum) {
    mYoloClassNum = yoloClassNum;
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
        return (IPluginV2*)(new UpSamplePlugin(2));
    }
    else if(strName.find("yolo-det") != std::string::npos) {
        return (IPluginV2*)(new YoloLayerPlugin(mYoloClassNum));
    }
    else
    {
        std::cout << "warning : " << layerName << std::endl;
        assert(0);
        return nullptr;
    }
}


