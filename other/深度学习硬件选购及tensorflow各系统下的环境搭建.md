# 一、深度学习硬件选购

- [Keras windows - Keras中文文档](https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_windows/)（含基本开发环境搭建、keras 框架搭建）



# 二、tensorflow环境搭建

## 1. 基本认识

1.显卡（GPU）是否支持 CUDN

<https://developer.nvidia.com/cuda-gpus>

2.了解基础知识

1）**CUDA（Compute Unified Device Architecture）**，是显卡厂商 NVIDIA 推出的运算平台。 CUDA™是一种由NVIDIA 推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。

计算行业正在从只使用 CPU 的“中央处理”向 CPU 与 GPU 并用的“协同处理”发展。为打造这一全新的计算典范，NVIDIA™（英伟达™）发明了 CUDA（Compute Unified Device Architecture，统一计算设备架构）这一编程模型，是想在应用程序中充分利用 CPU 和 GPU 各自的优点。现在，该架构已应用于GeForce™（精视™）、ION™（翼扬™）、Quadro 以及 Tesla GPU（图形处理器）上。

2）**cuDNN** 的全称为 NVIDIA CUDA® Deep Neural Network library，是 NVIDIA 专门针对深度神经网络（Deep Neural Networks）中的基础操作而设计基于 GPU 的加速库。基本上所有的深度学习框架都支持 cuDNN 这一加速工具，例如：Caffe、Caffe2、TensorFlow、Torch、Pytorch、Theano 等。

3）Anaconda 是一个开源的 Python 发行版本，其包含了 conda、Python 等 180 多个科学包及其依赖项。因为包含了大量的科学包，Anaconda 的下载文件比较大，如果只需要某些包，或者需要节省带宽或存储空间，也可以使用 Miniconda 这个较小的发行版（仅包含 conda 和 Python）。

——from：<https://www.cnblogs.com/chamie/p/8707420.html>



## 2. Windows下安装TensorFlow

### 需要下载

① NVIDIA 驱动程序下载地址：https://www.nvidia.cn/Download/index.aspx?lang=cn，进去会自动识别显卡型号

② CUDA 下载地址：https://developer.nvidia.com/cuda-toolkit-archive，如下（2019-06-21）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621191552.png)

③ cuDNN 的下载地址：https://developer.nvidia.com/rdp/cudnn-download，如下（2019-06-21，需要注册账号才能下载）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621192709.png)

点击 [Archived cuDNN Releases](https://developer.nvidia.com/rdp/cudnn-archive) 可以看到如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621192413.png)

注意1，担心在 windows 下安装软件出些幺蛾子，最好先安装好 **.Net Framework**。这是 .net framework 4.7.2 下载地址：[Download .NET Framework 4.7.2 | Free official downloads](https://dotnet.microsoft.com/download/dotnet-framework/net472)

注意2，GPU 显卡计算能力大于3.0才支持 cuDNN，查看 GPU 计算能力【https://developer.nvidia.com/cuda-gpus】

### 注意：版本问题

#### 第一点：

注意：要知道自己电脑的 CUDA 版本号，则可以选择合适版本的 **CUDA Toolkit**，例如下图的 CUDA 版本号为 9.2（如何查看参考：[Windows系统查看CUDA版本号](https://www.jianshu.com/p/d3b9419a0f89)）：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621193427.png)

则可以安装 CUDA Toolkit 9.2、CUDA Toolkit 9.0、CUDA Toolkit 9.1、CUDA Toolkit 8.0 等（我的理解：即安装的版本不能超过截图看到的版本。）

#### 第二点：

我们来看一篇文章的作者说的：

> **选择下载CUDA9.0**而不是CUDA10.0的原因：tensorflow_gpu库尚未发布与CUDA10.0对应的版本。本文作者写作此文时间是2018年11月14日，此时市面上tensorflow_gpu版本大多容易与CUDA9.0相配合。

说明也不能随便下载 cuda 的，需要根据你使用的 tensorflow-gpu 版本来决定下载哪个版本 cuda。

---

截止 2019-06-21 最新 tensorflow 版本为 2.0.0-alpha0。可以看到这里【[GPU 支持  |  TensorFlow](https://www.tensorflow.org/install/gpu?hl=zh-CN)】提到：

> TensorFlow 2.0 Alpha 可用于测试并支持 GPU。要进行安装，请运行以下命令：
>
> ``` python
> pip install tensorflow-gpu==2.0.0-alpha0
> ```

并且看到下面还写道，可以看到使用 tensorflow 2.0.0-alpha0 的要求：

> 必须在系统中安装以下 NVIDIA® 软件：
>
> - [NVIDIA® GPU 驱动程序](https://www.nvidia.com/drivers) - CUDA 10.0 需要 410.x 或更高版本。
> - [CUDA® 工具包](https://developer.nvidia.com/cuda-zone) - TensorFlow 支持 CUDA 10.0（TensorFlow 1.13.0 及更高版本）
> - CUDA 工具包附带的 [CUPTI](http://docs.nvidia.com/cuda/cupti/)。
> - [cuDNN SDK](https://developer.nvidia.com/cudnn)（7.4.1 及更高版本）
> - （可选）[TensorRT 5.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)，可缩短在某些模型上进行推断的延迟并提高吞吐量。

可以看到：

1. 安装 cuda 某个版本，那安装 NVIDIA 驱动版本也有要求
2. tensorflow 2.0.0-alpha0，支持 cuda 10.0，cuDNN 7.4.1 及以上。
3. tensorflow 1.13.0 及以上，都支持 cuda 10.0

补充，查看已经安装的驱动的版本方法，在【设备管理器】找到要查看的驱动，右键驱动【属性】，切换到【驱动程序】，可以看到如下截图：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190621205233.png)



#### 我个人的理解和小结（仅参考）

1、先打开你电脑的【NVIDA 控制面板】->【系统信息】->【组件】查看 `NVCUDA.DLL` 这项的产品名称，其中名称中的版本表示你安装的 CUDA 不能超过它。

2、然后根据你要安装的 tensorflow 版本所支持的 CUDA 版本。比如 `tensorflow-gpu 2.0.0-alpha0` 支持 CUDA 10.0

3、然后根据你要安装的 tensorflow 查看所支持的 cuDNN 版本，比如 `tensorflow-gpu 2.0.0-alpha0` 支持 cuDNN 7.4.1 及更高版本

4、选择出 cuDNN 版本后，然后进【https://developer.nvidia.com/rdp/cudnn-download】选择 cuDNN for CUDA 版本。

### 最后：命令安装 tensorflow

前面各个软件和工作的我就在此安装省略了。

windows 下安装：

- CPU版：`pip3 install --upgrade tensorflow`

- GPU版：`pip3 install --upgrade tensorflow-gpu`



#### 参考文章

- [win7 64位+CUDA 9.0+cuDNN v7.0.5 安装](https://blog.csdn.net/shanglianlm/article/details/79404703)  [荐] 
- [这是一份你们需要的Windows版深度学习软件安装指南](https://zhuanlan.zhihu.com/p/29903472)  [荐]
- [深度学习环境搭建-CUDA9.0、cudnn7.3、tensorflow_gpu1.10的安装](<https://www.jianshu.com/p/4ebaa78e0233>)



## 3. Ubuntu下安装TensorFlow



参考：

- [从零开始搭建深度学习服务器: 基础环境配置（Ubuntu + GTX 1080 TI + CUDA + cuDNN）](http://www.52nlp.cn/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E6%90%AD%E5%BB%BA%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AEubuntu-1080ti-cuda-cudnn)

