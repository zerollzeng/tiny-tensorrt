# 如何编写自定义插件

## 概述

TensorRT已经只支持了许多常见的神经网络层,比如卷积, 池化, BN等等. 但是依然还有很多操作和算子是不支持的,所以TensorRT提供了接口让我们可以编写插件来实现自己的自定义层. 这个接口有c++ 和 python版本,但是因为这个项目自己用pybind11实现了python绑定c++, 所以下面只以c++版本为基础进行介绍.

## 实现自定义插件需要的类

如果你查阅TensorRT的官方文档的话, 你会发现有IPlugin和IPluginV2两个插件的基础类, IPluginV2是5.0版本新加的升级版,所以IPlugin这个接口在未来会被移除. 如果要编写自定义插件的话, 还是使用IPluginV2,本文的介绍都是针对IPluginV2的.

IPluginV2是一个基类, 还有一些派生类,它们可以提供更加丰富的功能.

![image](https://user-images.githubusercontent.com/38289304/69928212-f3ea8f00-14f5-11ea-9b8e-630fb367cf59.png)

| |Introduced in TensorRT version | Mix input/output formats/type | Dynamic shapes | Requires extended runtime |
| :-: | :-: | :-: | :-: | :-: |
| IPluginV2Ext | 5.1 | Limited | No | No |
| IPluginV2IOExt | 6.0.1 | General | No | No |
| IPluginV2DynamicExt | 6.0.1 | General | Yes | Yes |

要编写自己的自定义层,就是要定义一个自定义层的类, 这个类继承自上述任意一个基类.官方推荐是继承IPluginV2IOExt或者IPluginV2DynamicExt,但是我的看法是还是根据自己的需求来定, 在满足要求的情况下优先先继承更顶层的类,后续再根据实际需求再决定是否需要升级. 毕竟下面的类也是在IPluginV2的基础上添加新的feature.


这是一个最基础的继承自IPluginV2的自定义插件的头文件,所有自定义插件都至少需要实现下面所有的virtual方法,如果继承了具备更多特性的库,那么还需要实现其他需要要求的方法.你可以先浏览一下下面的函数名,接下来我会介绍这些函数的调用工作流.
```
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

### workflow

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


