* Self-Attention ConvLSTM for Spatiotemporal Prediction
* `SaConvSTLM` described in this [paper](https://ojs.aaai.org//index.php/AAAI/article/view/6819)
* test on `MovingMNIST` [dataset](http://www.cs.toronto.edu/%7Enitish/unsupervised_video/mnist_test_seq.npy)



## project structure

* `load_data.py`

  包含用于下载数据集、对数据集进行预处理和保存预测图片到文件夹的实用函数

* `ConvSTLM_main.py`

  使用 ·`tensorflow.keras.layers.ConvLSTM2D` 搭建的网络模型及其训练测试

* `SaConvSTLM.py`

  基于`tensorflow`实现了cell `SaConvLSTM2DCell`和 layer `SaConvLSTM2D`，layer`SaConvLSTM2D`可以直接作为模型中的一个 layer 使用

* `SaConvSTLM_main.py`

  使用 自定义的`SaConvLSTM`层搭建的网络模型及其在`MovingMNIST` 上的训练测试

  **模型结构**

![image-20210317230847173](/Users/zhangyouyi/Library/Mobile Documents/com~apple~CloudDocs/Typora/RelyFiles/image-20210317230847173.png)

## `SaConvLSTM` structure

* described in this [paper](https://ojs.aaai.org//index.php/AAAI/article/view/6819)

* overall pattern

  ![image-20210317231218553](/Users/zhangyouyi/Library/Mobile Documents/com~apple~CloudDocs/Typora/RelyFiles/image-20210317231218553.png)

* self_attention memory module

![image-20210317231301247](/Users/zhangyouyi/Library/Mobile Documents/com~apple~CloudDocs/Typora/RelyFiles/image-20210317231301247.png)
