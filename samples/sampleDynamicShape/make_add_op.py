import onnx
import numpy as np
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

x = onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [4,128,-1,-1])
y = onnx.helper.make_tensor_value_info("y", TensorProto.FLOAT, [4,128,-1,-1])

z = onnx.helper.make_tensor_value_info("z", TensorProto.FLOAT, [4,128,-1,-1])

node = onnx.helper.make_node("Add", ["x","y"], ["z"])

graph = onnx.helper.make_graph([node], "Add", [x, y], [z])

model_def = onnx.helper.make_model(graph, producer_name="add_def")

onnx.checker.check_model(model_def)

onnx.save_model(model_def, "/tmp/sample_add.onnx")
