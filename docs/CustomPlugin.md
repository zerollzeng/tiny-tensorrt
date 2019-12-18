English | [中文简体](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/CustomPlugin-CN.md)

# How to write your custom plugin

## Overview

TensorRT already support most of common neural network layer such as convolution, pooling and BN, in practical deployment, there are still a lot of operation that it doesn't support. So TensorRT provide interface which we can write our custom plugin for support our custom layer. the remains of this article I will introduce how to write your custom plugin in c++, since tiny-tensorrt support python with pybind11, so if you use python interface, you still need to write your custom plugin in c++. after reading this tutorial. you can use plugin/PReLUPlugin as your template, and write your implementation according to my comments.

## Class we need

## nvinfer1::IPluginV2/IPluginV2Ext/IPluginV2IOExt/IPluginV2DynamicExt

If you read TensorRT's official documentation, you can find IPlugin and IPluginV2 classes. since IPluginV2 was added in 5.0 version, and IPlugin will be deprecated in the future. so if you want to write your custom plugin, I recommend you use IPluginV2 instead of IPlugin.

IPluginV2 is a basic class, there are other classes that support more features

![image](https://user-images.githubusercontent.com/38289304/69928212-f3ea8f00-14f5-11ea-9b8e-630fb367cf59.png)

| |Introduced in TensorRT version | Mix input/output formats/type | Dynamic shapes | Requires extended runtime |
| :-: | :-: | :-: | :-: | :-: |
| IPluginV2Ext | 5.1 | Limited | No | No |
| IPluginV2IOExt | 6.0.1 | General | No | No |
| IPluginV2DynamicExt | 6.0.1 | General | Yes | Yes |

Write your custom plugin, means define a class which inherit from one of base class list above. implement all of the virtial method, if you read the official developer's guide, it recommend you inherit from IPluginV2IOExt or IPluginV2DynamicExt, but in my opinion, I suggest you depend on your requirements. if IPluginV2 meets your requirements, them just use IPluginV2, you can upgrade to other class anytime you want, it will reduces the time you spend.

This is an example header file which I define a CustomPlugin class inherit from IPluginV2, you have to implement all the virtual methods, if you inherit from IPluginV2Ext or IPluginV2IOExt, you have to implement their virtual methods.

```c++
class CustomPlugin : public nvinfer1::IPluginV2
{
public:
    CustomPlugin(const Weights *weights, int nbWeights);

    CustomPlugin(const void *data, size_t length);

    ~CustomPlugin();

    virtual int getNbOutputs() const override;

    virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    virtual bool supportsFormat(DataType type, PluginFormat format) const override;

    virtual void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;
    
    virtual int initialize() override;

    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int maxBatchSize) const override;
    
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
    
    virtual size_t getSerializationSize() const override;

    virtual void serialize(void* buffer) const override;

    virtual const char* getPluginType() const override;
    
    virtual const char* getPluginVersion() const override;
    
    virtual void destroy();
    
    virtual IPluginV2* clone() const override;

    virtual void setPluginNamespace(const char* pluginNamespace) override;
    
    virtual const char* getPluginNamespace() const override;
};
```

### Workflow of IPluginV2 plugin

#### Parse phase

In parse phase tensorrt will create every instance of custom plugin of your model, and get output counts and dimensions of your custom layer by getNbOputputs() and getOutputDimensions(), for build the whole workflow of your model, if the output counts and dimensions do not match the next layer, will bring parse failure. so if your model parse fails, you can check this two function, see if they return correct output counts and dimensions.

#### Build engine phase

At engine building phase, tensorrt will call supportFormat() check the support formats of your custom plugin, it depend on your implementation. and when building the engine, tensorrt will call configureWithFormat(), according you configuration to set the plugin with proper datatype and plugin format.also at this phase, will call getWorkspaceSize() which is not important at all. and at last, will call initialize() to initialize your plugin.when finish initialize, your custom plugin is ready for execution. when you call destroy() of builder,network or engine, they will call plugin's destroy() and destruct the plugin. 

#### Save engine phase

For save engine, tensorrt will call getSerializationSize() and serialize() to get size it need for serialize and serialize your custom plugin to engine file

#### Run engine phase

Will call enqueue()

#### Infer with engine file

When deserialize and infer with deserialized engine, at first it will call SamplePlugins(const void *data, size_t length) deserialize the plugin from buffer, and initialize with initialize(), and call enqueue when infer. and when all infer was done, call terminate() to release resources.

## nvinfer1::IPluginCreator

IPluginCreator register your plugin to plugin registry, when you use custom plguin with uff model or deserialize from engien file, you need IPluginCreator to get your custom plugin. belows are methods of IPluginCreator, for details please refer sample code.
```c++
class CustomPluginCreator : public nvinfer1::IPluginCreator {
public:
    CustomPluginCreator();

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
};

```
if you write plugin for uff model, IPluginV2 will call getFieldNames() and createPlugin(const char* name, const nvinfer1::PluginFieldCollection *fc) to get instance for your custom plugin, so if you only need it for caffe, you can ignore those two methods and return nullptr for them since caffe plugin was create through PluginFactory describe below. and when you do inference from deserialize engine file,  deserializePlugin(const char* name, const void* serialData, size_t serialLenth) is necessary so your must implement this method. for details please refer sample code.

## nvcaffeparser1::IPluginFactoryV2

this class is use for caffe model, just refer to plugin/PluginFactory.h and plugin/PluginFactory.cpp.

this class is very similar to IPluginCreator, the difference is for every you need to implement their IPluginCreator separately and only need one PluginFactory for all of them(now you get why it call Factory lol)

## Sample code
please refer plugin/PReLUPlugin and plugin/plugin_utils.h/cpp they are well_documented and you can just use it as template.