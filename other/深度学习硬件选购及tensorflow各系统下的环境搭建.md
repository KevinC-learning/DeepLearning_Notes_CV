## 1. 深度学习硬件选购

- [Keras windows - Keras中文文档](https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_windows/)（含基本开发环境搭建、keras 框架搭建）



## 2. tensorflow环境搭建

### (0) 基本认识

1.显卡（GPU）是否支持 CUDN

<https://developer.nvidia.com/cuda-gpus>

2.了解基础知识

1）CUDA（Compute Unified Device Architecture），是显卡厂商 NVIDIA 推出的运算平台。 CUDA™是一种由NVIDIA 推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。

计算行业正在从只使用 CPU 的“中央处理”向 CPU 与 GPU 并用的“协同处理”发展。为打造这一全新的计算典范，NVIDIA™（英伟达™）发明了 CUDA（Compute Unified Device Architecture，统一计算设备架构）这一编程模型，是想在应用程序中充分利用 CPU 和 GPU 各自的优点。现在，该架构已应用于GeForce™（精视™）、ION™（翼扬™）、Quadro 以及 Tesla GPU（图形处理器）上。

2）cuDNN 的全称为 NVIDIA CUDA® Deep Neural Network library，是 NVIDIA 专门针对深度神经网络（Deep Neural Networks）中的基础操作而设计基于 GPU 的加速库。基本上所有的深度学习框架都支持 cuDNN 这一加速工具，例如：Caffe、Caffe2、TensorFlow、Torch、Pytorch、Theano 等。

3）Anaconda 是一个开源的 Python 发行版本，其包含了 conda、Python 等 180 多个科学包及其依赖项。因为包含了大量的科学包，Anaconda 的下载文件比较大，如果只需要某些包，或者需要节省带宽或存储空间，也可以使用 Miniconda 这个较小的发行版（仅包含 conda 和 Python）。

——from：<https://www.cnblogs.com/chamie/p/8707420.html>



### (1) Windows 下安装 TensorFlow

CPU版：`pip3 install --upgrade tensorflow`

GPU版：`pip3 install --upgrade tensorflow-gpu`

参考：

- [win7 64位+CUDA 9.0+cuDNN v7.0.5 安装](https://blog.csdn.net/shanglianlm/article/details/79404703)  [荐] 
- [这是一份你们需要的Windows版深度学习软件安装指南](https://zhuanlan.zhihu.com/p/29903472)
- [深度学习环境搭建-CUDA9.0、cudnn7.3、tensorflow_gpu1.10的安装](<https://www.jianshu.com/p/4ebaa78e0233>)



### (2) Ubuntu 下安装 TensorFlow



参考：

- [从零开始搭建深度学习服务器: 基础环境配置（Ubuntu + GTX 1080 TI + CUDA + cuDNN）](http://www.52nlp.cn/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%90%AD%E5%BB%BA%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AEubuntu-1080ti-cuda-cudnn)

