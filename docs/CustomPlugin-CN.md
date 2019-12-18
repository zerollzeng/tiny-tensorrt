[English](https://github.com/zerollzeng/tiny-tensorrt/blob/master/docs/CustomPlugin.md) | 中文简体

# 如何编写自定义插件

## 概述

TensorRT已经只支持了许多常见的神经网络层,比如卷积, 池化, BN等等. 但是依然还有很多操作和算子是不支持的,所以TensorRT提供了接口让我们可以编写插件来实现自己的自定义层. 这个接口有c++ 和 python版本,但是因为这个项目自己用pybind11实现了python绑定c++, 所以下面只以c++版本为基础进行介绍. 要编写你的自定义插件,可以使用plugin/PReLUPlugin内文件作为模板,他们都有非常详细的注释.只需要按照我的注释去实现即可.

## 实现自定义插件需要的类

-----

## nvinfer1::IPluginV2/IPluginV2Ext/IPluginV2IOExt/IPluginV2DynamicExt

如果你查阅TensorRT的官方文档的话, 你会发现有IPlugin和IPluginV2两个插件的基础类, IPluginV2是5.0版本新加的升级版,所以IPlugin这个接口在未来会被移除. 如果要编写自定义插件的话, 还是使用IPluginV2,本文的介绍都是针对IPluginV2的.

IPluginV2是一个基类, 还有一些派生类,它们可以提供更加丰富的功能.

![image](https://user-images.githubusercontent.com/38289304/69928212-f3ea8f00-14f5-11ea-9b8e-630fb367cf59.png)

| |TensorRT版本 | 混合精度 | 动态大小输入 | Requires extended runtime |
| :-: | :-: | :-: | :-: | :-: |
| IPluginV2Ext | 5.1 | Limited | No | No |
| IPluginV2IOExt | 6.0.1 | General | No | No |
| IPluginV2DynamicExt | 6.0.1 | General | Yes | Yes |

要编写自己的自定义层,就是要定义一个自定义层的类, 这个类继承自上述任意一个基类.官方推荐是继承IPluginV2IOExt或者IPluginV2DynamicExt,但是我的看法是还是根据自己的需求来定, 在满足要求的情况下优先先继承更顶层的类,后续再根据实际需求再决定是否需要升级. 毕竟下面的类也是在IPluginV2的基础上添加新的feature.


这是一个最基础的继承自IPluginV2的自定义插件的头文件,所有自定义插件都至少需要实现下面所有的virtual方法,如果继承了具备更多特性的库,那么还需要实现其他需要要求的方法.你可以先浏览一下下面的函数名,接下来我会介绍这些函数的调用工作流.
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

### IPluginV2插件的工作流

#### parse phase/ parse阶段

在模型的parse阶段会通过CustomPlugin(const Weights *weights, int nbWeights)创建模型中每一个自定义层的实例, 在这个阶段还会调用到getNbOutputs()和getOutputDimensions()来获取自定义层的输出个数和维度, 这个步骤的目的是为了构建整一个模型的工作流.如果自定义层的输出个数和维度跟其他层匹配不上,parse就会失败.所以如果你的自定义层在parse阶段就parse失败了,可以先检查一下这两个函数的实现. 这个阶段创建的CustomPlugin实例会在engine构建阶段被析构掉.

#### build engine phase / engine构建阶段

engine构建阶段会再次通过CustomPlugin(const Weights *weights, int nbWeights)创建自定义层的实例.然后调用supportFormat()函数来检查自定义层的支持的Datatype和PluginFormat, 在build的过程中,会调用configureWithFormat,根据设定的类型(见参数)对插件进行配置.调用完这个函数之后,自定义层内部的状态和变量应该被配置好了.在这里也会调用getWorksapceSize(),但是这个函数不怎么重要.最后会调用initialize(),进行初始化.此时已经准备好所有准备的数据和参数可以进行执行了.构建结束后当调用builder, network或者 engine的destroy()函数时,会调用CustomPlugin的destroy()方法析构掉CustomPlugin对象.

#### save engine phase / 引擎保存阶段

保存引擎到序列化文件会调用getSerializationSize()函数来获取序列化所需要的空间,在保存的过程中会调用serialize()函数将自定义层的相关信息序列化到引擎文件.

#### engine running phase / 引擎推理阶段

在这个阶段会调用用enqueue()进行模型推理

#### inference with engine file / 使用引擎文件进行推理

在使用引擎文件进行推理的过程中,从序列化文件恢复权重和参数,所以会先调用SamplePlugins(const void *data, size_t length)读取自定义层的相关信息,然后调用initialize() 进行初始化.在推理的过程中调用enqueue()进行推理.推理结束后如果在调用engine的destroy方法的时候会调用terminate()函数,释放
掉initialize()申请的资源.

-----

## nvinfer1::IPluginCreator

IPluginCreator主要用于将编写好的IPlugin插件注册到Plugin Registry, 在解析uff(tensorflow 模型)的时候就可以调用到自定义层的IPluginV2实现,以及在反序列化engine文件的时候也会通过IPluginCreator来获取自定义层.这里是IPluginCreator的函数的方法.更具体的参见示例文件即可.
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

这里主要要介绍的就是如果要编写uff的自定义插件,IPluginCreator会通过getFieldNames() 和 createPlugin(const char* name, const nvinfer1::PluginFieldCollection *fc) 来获取自定义插件的实例,所以如果你只需要caffe模型的自定义插件,这两个函数不实现返回nullptr即可.但是 caffe模型的自定义插件需要实现下一节的nvcaffeparser1::IPluginFactoryV2接口. deserializePlugin(const char* name, const void* serialData, size_t serialLenth) 是在反序列engine的时候调用的,所以这个函数必须要实现.具体参见示例文件.

最后在cpp文件的最后,不要忘记加上REGISTER_TENSORRT_PLUGIN(pluginCreator)这个宏来注册你的自定义插件

-----

## nvcaffeparser1::IPluginFactoryV2

这个类是caffe模型专用的,主要通过这个类来创建caffe的自定义插件.请直接参见plugin/PluginFactory.h和plugin/PluginFactory.cpp,只需要模仿我的实现在对应的函数内把你的自定义插件的相关信息添加进去即可.

这个类的用途跟IPluginCreator非常相似,区别就是每一个plugin都需要实现一个自己的IPluginCreator,而PluginFactory只需要一个(所以叫做工厂哈哈).


## 示例文件

请参见plugin/PReLUPlugin内文件和plugin/plugin_utils.h以及plugin/plugin_utils.cpp,他们都有非常详细的注视,只需要按照模板实现即可~


