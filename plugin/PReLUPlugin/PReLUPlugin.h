/*
 * @Email: zerollzeng@gmail.com
 * @Author: zerollzeng
 * @Date: 2019-12-02 16:31:56
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-12-11 14:38:17
 */
#ifndef PRELU_PLUGIN_H // modidy to your file name
#define PRELU_PLUGIN_H // 改成你自己的文件名

/**
 * @description: this is a plugin sample code with detailed explaination,
 *               you can copy and modify base on this file. when reading
 *               this header file, please also read the PReLUPlugin.cpp,
 *               if something is unclear to you, feel free to bring me an
 *               issue:)
 * @描述: 这是一份示例代码,你可以直接复制,然后在上面根据你自己的插件修改,你应该结合
 *       PReLUPlugin.cpp来理解注释,如果有不清晰的地方,欢迎给我提issue
 * @warning: before you write your own layer, make sure you are very familiar 
 *           to the detialed implementation of custom layer, how it compute,
 *           how many weights does it have.
 * @警告: 在实现自定义层之前,请先确保你已经足够熟悉要实现层的内部运作方式,有多少参数,很多莫名
 *       其妙的bug都来源于对这个的不熟悉,而且这是一个很好的学习机会~
 */

// header file

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "spdlog/spdlog.h"


class PReLUPlugin : public nvinfer1::IPluginV2
{
public:
    /**
     * @description: construction
     * @描述: 构造函数
     * @param weights and nbWeight is the parameters of PluginFactory::createPlugin,
     *        see PluginFactory.cpp, if your costom layer has not any weights in
     *        model, delete this two parameter is ok. and if your custom layer has
     *        has other parameters does not in model weights file, you should add
     *        your own parameter, see UpsamplePlugin. This function is mainly read
     *        the plugin's weights or other parameter to internal variable.
     * @参数: weights 和 nbWeight这两个参数是PluginFactory::createPlugin的参数,可以参
     *       可以参见PluginFactory.cpp, 如果你的自定义层没有权重,那么这两个参数你不要也可以
     *       如果有额外的参数,也要把额外的参数添加进来,关于这个可以参考UpsamplePlugin的实现 .
     *       这个函数主要就是用来将权重和自定义层的其他参数读取到内部变量里面.
     */
    PReLUPlugin(const nvinfer1::Weights* weights, int nbWeight);

    /**
     * @description: create an plugin from serialized data, it's a counter part 
     *               of serialize(), which write the class information such as 
     *               private member variable to serialized data. this fuction will
     *               be call by IPluginCreator::deserializePlugin(), note that 
     *               the order of read must be same with write.
     * @描述: 这个就是从序列化数据里面恢复plugin的相关数据,另一个函数serialize将类的数据写
     *       入到序列化数据里面.在IPluginCreator::deserializePlugin里面会调用到这个函数,
     *       注意写的顺序跟读的顺序必须是一样的.
     */
    PReLUPlugin(const void* data, size_t length);

    /**
     * @description:return how many space when serialize your custom plugin, include
     *              weights and necessary member variable. because when restore
     *              engine from serilized engine file, it won't call configureWithPlugin
     *              method.
     * @描述: 返回在序列化你的自定义插件的时候,需要占用到多少空间,其实就是你的权重和一些必要的成员变量.
     *       因为在使用序列化引擎的时候,不会调用configureWithPlugin函数,第一次配置好的信息需要
     *       你自己保存到序列化文件.
     */
    virtual size_t getSerializationSize() const override;

    /**
     * @description: serialize you custom plugin to buffer, include weights and
     *               necessary member variable, for this you can mimic my implementation,
     *               noted that the order of write is the same with read. see
     *               PReLUPlugin(const void* data, size_t length);
     * @描述: 序列化你的自定义插件到buffer,你可以直接模仿我的实现,只需要保证write的顺序和read的顺序是
     *       一样的,不然在反序列化的时候就会得到错误的值,参考PReLUPlugin(const void* data, size_t length);
     */
    virtual void serialize(void* buffer) const override;

    /**
     * @description: no params construction has no mean, so delete it
     * @描述: 无参构造函数没有意义
     */
    PReLUPlugin() = delete;

    /**
     * @description: destructiion, free resource
     * @描述:析构函数,释放资源
     */
    ~PReLUPlugin();

    /**
     * @description: return the number of output tensors. for prelu return 1 the
     *               same as relu. it depends on your custom layer
     * @描述: 返回输出tensor的数量, 比如说prelu,输出个数跟relu一样是1,这个取决于你的自定义层.
     */
    virtual int getNbOutputs() const override;
    
    /**
     * @description: return dimensions of output tensor, this might depends on 
     *               input tensors dimensions. for prelu, the output dimension 
     *               is the same as input dimension.
     * @描述 返回输出tensor的维度,很多时候都取决于输入维度.对于prelu来说,输出维度等于输入维度.
     * @param index	The index of the output tensor.
     * @参数 index 输出tensor的index
     * @param inputs	The input tensors.
     * @参数 inputs 输出tensors的纬度.注意有可能有多个输入
     * @param nbInputDims	The number of input tensors.
     * @参数 nbInputDims 输出tensors的个数.
     */
    virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    /**
     * @description: query whether a given datatype and plugin format is support,
     *               it depends on your custom layer implementation.
     * @描述 查询对应的datatype和format是否支持, 这个取决于你的自定义层实现是否支持.
     */
    virtual bool supportsFormat(const nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;

    virtual void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
                                     int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format,
                                     int maxBatchSize) override;

    /**
     * @description: initialize your plugin for execution, for simplicity, you need
     *               to prepare data in gpu in this function, for example convert
     *               and copy your weights to gpu.because after this method, enqueue
     *               will be call.
     * @描述: 初始化你的插件,其实就是初始化好gpu context, 将你的权重从内存拷贝到gpu上,如果
     *        设定了fp16,当然也要先做转化在拷贝到gpu.
     * @return: 
     */
    virtual int initialize() override;

    /**
     * @description: free memory, include cpu and gpu, see cpp file.
     * @描述: 释放内存和显存, 见cpp
     */
    virtual void terminate() override;

    /**
     * @description: hard to explain, just return 0;
     * @描述: 很难解释, 直接返回0即可.
     */
    virtual size_t getWorkspaceSize(int maxBatchSize) const override;

    /**
     * @description: see cpp
     * @描述: 见cpp
     */
    virtual const char* getPluginType() const override;

    /**
     * @description: see cpp
     * @描述: 见cpp
     */
    virtual const char* getPluginVersion() const override;

    /**
     * @description: the same as ~PReLUPlugin(),just copy my implementation.
     * @描述: 调用这个接口来析构,参考我的代码即可.
     */
    virtual void destroy();

    /**
     * @description: see cpp
     * @描述: 见cpp
     */
    virtual nvinfer1::IPluginV2* clone() const override;

    /**
     * @description: DO NOT IMPLEMENT THIS FUNCTION
     * @描述: 不要实现这个方法,留空即可
     */
    virtual void setPluginNamespace(const char* pluginNamespace) override {}

    /**
     * @description: see cpp
     * @描述: 见cpp
     */
    virtual const char* getPluginNamespace() const override;
    
    /**
     * @description: see cpp
     * @描述: 见cpp
     */
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs,
                        void* workspace, cudaStream_t stream) override;

private:
    int mNbInputChannels, mNbInputHeight, mNbInputWidth;
    nvinfer1::Weights mWeights;
    nvinfer1::DataType mDataType{nvinfer1::DataType::kFLOAT};
    void* mDeviceKernel{nullptr};
};

/**
 * @description: see cpp and mimic my implementation
 * @描述: 直接参见cpp并且直接模仿我的实现即可.
 */
class PReLUPluginCreator : public nvinfer1::IPluginCreator {
public:
    PReLUPluginCreator();

    // ------------------inherit from IPluginCreator-------------------
    // return the plugin type + plugin namesapce
    virtual const char* getPluginName() const override;

    // return the plugin version
    virtual const char* getPluginVersion() const override;

    // return a list of fields that needs to be passed to createPlugin
    virtual const nvinfer1::PluginFieldCollection* getFieldNames() override;

    // return nullptr in case of error
    virtual nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection *fc) override;

    // Called during deserialization of plugin layer. Return a plugin object.
    virtual nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLenth) override;

    // Set the namespace of the plugin creator based on the plugin library it belongs to. This can be set while registering the plugin creator
    virtual void setPluginNamespace(const char* pluginNamespace) override {}

    // Return the namespace of the plugin creator object.
    virtual const char* getPluginNamespace() const override;

private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
};

#endif //PLGUIN_SAMPLE_H