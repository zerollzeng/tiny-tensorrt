Note: This template runs successfully with TensorRT 7.2.2 and failed at some previous versions.

## A Cute custom onnx plugin template

Want to implement a custom plugin for your onnx model and don't know where to start? here is a simple template, let's go!

### Step 1

copy this whole CuteSamplePlugin directory to your project's plugin directory, or, as I recommended, implement it with tiny-tensorrt, then port to your own project.

### Step 2

Now check the name of your node that need to be registered as a plugin, you can find its name via [Netron](https://netron.app/), after import your model, you should see your node name here

![image](https://user-images.githubusercontent.com/38289304/104086586-41f2ca00-5294-11eb-8bb6-af2f127908b2.png)

now I get the Plugin name(which is the type attribute of the node) is HSigmoid, open CuteSamplePlugin.cpp, find

```c++
static const char* CUTE_PLUGIN_VERSION{"1"};
static const char* CUTE_PLUGIN_NAME{"CuteSamplePlugin"};
```

edit the CUTE_PLUGIN_NAME to

```c++
static const char* CUTE_PLUGIN_NAME{"HSigmoid"};
```

### Step 3

Now if you run it with your own plugin that contains the custom plugin node, it should find your custom plugin(take a look at test/test.cpp). you should get a running output like this

```
root@f44d36162a5e:/tiny-tensorrt/build# ./unit_test --onnx_path ../demo.onnx 
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][CuteSamplePluginCreator][Line 155] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getPluginName][Line 162] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getPluginName][Line 162] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getPluginVersion][Line 168] wow I run to here now
[2021-01-09 07:50:56.784] [info] create plugin factory
[2021-01-09 07:50:56.784] [info] yolo3 params: class: 1, netSize: 416 
[2021-01-09 07:50:56.784] [info] upsample params: scale: 2
[2021-01-09 07:50:56.827] [info] build onnx engine from ../demo.onnx...
----------------------------------------------------------------
Input filename:   ../demo.onnx
ONNX IR version:  0.0.4
Opset version:    9
Producer name:    pytorch
Producer version: 1.3
Domain:           
Model version:    0
Doc string:       
----------------------------------------------------------------
/workspace/TensorRT/parsers/onnx/ModelImporter.cpp:139: No importer registered for op: HSigmoid. Attempting to import as plugin.
/workspace/TensorRT/parsers/onnx/builtin_op_importers.cpp:3716: Searching for plugin: HSigmoid, plugin_version: 1, plugin_namespace: 
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getPluginVersion][Line 168] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getFieldNames][Line 174] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][createPlugin][Line 180] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][CuteSamplePlugin][Line 32] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][setPluginNamespace][Line 143] wow I run to here now
/workspace/TensorRT/parsers/onnx/builtin_op_importers.cpp:3733: Successfully created plugin: HSigmoid
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getNbOutputs][Line 43] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getOutputDataType][Line 67] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][CuteSamplePlugin][Line 32] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][setPluginNamespace][Line 143] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getNbOutputs][Line 43] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getOutputDataType][Line 67] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][isOutputBroadcastAcrossBatch][Line 94] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getOutputDimensions][Line 49] wow I run to here now
 -------> (Unnamed Layer* 0) [Constant]_output 
x (Unnamed Layer* 0) [Constant]_output  -------> 2 
[2021-01-09 07:50:56.828] [info] fp16 support: false
[2021-01-09 07:50:56.828] [info] int8 support: false
[2021-01-09 07:50:56.828] [info] Max batchsize: 4
[2021-01-09 07:50:56.828] [info] Max workspace size: 10485760
[2021-01-09 07:50:56.828] [info] Number of DLA core: 0
[2021-01-09 07:50:56.828] [info] Max DLA batchsize: 268435456
[2021-01-09 07:50:56.828] [info] Current use DLA core: 0
[2021-01-09 07:50:56.828] [info] build engine...
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][CuteSamplePlugin][Line 32] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][setPluginNamespace][Line 143] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getPluginType][Line 119] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getPluginType][Line 119] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getPluginType][Line 119] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][supportsFormat][Line 113] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][supportsFormat][Line 113] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][supportsFormat][Line 113] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][supportsFormat][Line 113] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][supportsFormat][Line 113] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getOutputDataType][Line 67] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getOutputDataType][Line 67] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getOutputDataType][Line 67] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getOutputDataType][Line 67] wow I run to here now
Detected 1 inputs and 1 output network tensors.
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][CuteSamplePlugin][Line 32] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][setPluginNamespace][Line 143] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][configurePlugin][Line 108] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getWorkspaceSize][Line 61] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getWorkspaceSize][Line 61] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getWorkspaceSize][Line 61] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][getWorkspaceSize][Line 61] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][initialize][Line 55] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][destroy][Line 130] wow I run to here now
[2021-01-09 07:50:56.845] [info] serialize engine to 
[2021-01-09 07:50:56.845] [warning] empty engine file name, skip save
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][destroy][Line 130] wow I run to here now
[2021-01-09 07:50:56.845] [info] create execute context and malloc device memory...
[2021-01-09 07:50:56.845] [info] init engine...
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][CuteSamplePlugin][Line 32] wow I run to here now
 (๑¯◡¯๑) CUSTOM PLUGIN TRACE----> call [/tiny-tensorrt/plugin/CuteSamplePlugin/CuteSamplePlugin.cpp][setPluginNamespace][Line 143] wow I run to here now
[2021-01-09 07:50:56.846] [info] malloc device memory
nbBingdings: 2
[2021-01-09 07:50:56.846] [info] input: 
[2021-01-09 07:50:56.846] [info] binding bindIndex: 0, name: x, size in byte: 432
[2021-01-09 07:50:56.846] [info] binding dims with 4 dimemsion
1 x 3 x 3 x 3   
[2021-01-09 07:50:56.847] [info] output: 
[2021-01-09 07:50:56.847] [info] binding bindIndex: 1, name: 2, size in byte: 0
[2021-01-09 07:50:56.847] [info] binding dims with 4 dimemsion
1 x 0 x 0 x 0   
Out of memory
```

When the model runs forward, the plugin crash, that's because it doesn't contain the right implementation. But at least TenosrRT find your plugin. now continue your work until you get the right output of your model. Good Luck!

### Extra Help

if you are new to Nvidia TensorRT or don't know how to implement a plugin from scratch, you can take a look at

[TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)

official API documentation is very helpful

[TensorRT C++ API](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html)

official plugin implementation is also helpful if you are confused about some member function implementation.

[Official Plugin](https://github.com/NVIDIA/TensorRT/tree/master/plugin)
