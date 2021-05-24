import numpy as np
from PIL import Image
import os

# your calibrate data
img_list = {
    "/usr/src/tensorrt/data/resnet50/binoculars.jpeg",
    "/usr/src/tensorrt/data/resnet50/reflex_camera.jpeg",
    "/usr/src/tensorrt/data/resnet50/tabby_tiger_cat.jpg",
}

# save dir
save_dir = "./calibrate_data/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# you can get name from logs of tinyexec --onnx your_model.onnx
input_binding_name = "gpu_0/data_0"

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

i = 1
for img_path in img_list:
    img = Image.open(img_path)
    # match the input size of your onnx model
    img = img.resize((224,224))
    # you can add more image manipulation like padding, contrast adust
    # as you want

    np_img = np.asarray(img)
    np_img = np_img.transpose(2,0,1)
    np_img = preprocess(np_img)
    np_img = np.expand_dims(np_img, axis=0)
    print(np_img.shape)
    # add more numpy manipulation as you want

    save_path = save_dir + "%05d" % i + ".npz"
    dct = {input_binding_name: np_img}
    np.savez(save_path, **dct)
    i = i+1
