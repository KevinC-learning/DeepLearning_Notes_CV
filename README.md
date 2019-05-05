<a name="top"></a>

# TensorFlow学习记录


## 1. 前言

主要来源和参考的资料为：炼数成金视频《深度学习框架TensorFlow学习与应用》以及补充了一些网上的博文内容。

视频目录：

``` xml
第 1周 Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装
第 2周 Tensorflow的基础使用，包括对图(graphs),会话(session),张量(tensor),变量(Variable)的一些解释和操作
第 3周 Tensorflow线性回归以及分类的简单使用
第 4周 softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍
第 5周 卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题
第 6周 使用Tensorboard进行结构可视化，以及网络运算过程可视化
第 7周 递归神经网络LSTM的讲解，以及LSTM网络的使用
第 8周 保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别
第 9周 Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别
第10周 使用Tensorflow进行验证码识别
第11周 Tensorflow在NLP中的使用(一)
第12周 Tensorflow在NLP中的使用(二)
```

> 说明：实际第 5 周讲的是 Tensorborad 结构可视化，第 6 周讲的是 CNN，下载链接里的文件夹顺序，我已修正。

(1) 在线观看：

- YouTube：[tensorflow教程（十课）](https://www.youtube.com/watch?v=eAtGqz8ytOI&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk&index=2&t=0s)
- 或 B 站：[《深度学习框架TensorFlow学习与应用》](https://www.bilibili.com/video/av20542427/)

(2) 下载：

- 《深度学习框架Tensorflow学习与应用》（含视频+代码+课件，视频总时长：13小时31分钟）  【[百度网盘下载](https://pan.baidu.com/s/16OINOrFiRXbqmqOFjCFzLQ )  密码: 1a8j】
- 《深度学习框架Tensorflow学习与应用[只有videos-720p]》（该份资料只有视频文件） 【 [百度网盘下载](https://pan.baidu.com/s/1oQLgWFEBsVrcKJN4swEdzg)  密码: i3e2】

*其他学习视频（觉得有必要可以看看~）：*  

- 油管视频：[TF Girls 修炼指南](https://www.youtube.com/watch?v=TrWqRMJZU8A&list=PLwY2GJhAPWRcZxxVFpNhhfivuW0kX15yG&index=2)  或 B 站观看： [TF Girls 修炼指南](https://space.bilibili.com/16696495/#/channel/detail?cid=1588) 
- 油管视频：51CTO视频 [深度学习框架-Tensorflow案例实战视频课程](https://www.youtube.com/watch?v=-pYU4ub7g0c&list=PL8LR_PrSuIRhpEYA3sJ-J5hYGYUSwZwdS)、或 B 站观看：[深度学习框架-Tensorflow案例实战视频课程](https://www.bilibili.com/video/av29663946/?p=1)
- [Tensorflow 教程系列 | 莫烦Python](<https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/>)

(3) 相关资料：

- 郑泽宇/顾思宇：[《Tensorflow：实战Google深度学习框架》](https://book.douban.com/subject/26976457/) 出版时间 2017-2-10
  - 官方维护的书中的 TensorFlow 不同版本的示例程序仓库：<https://github.com/caicloud/tensorflow-tutorial>；
  - GitHub 有人写了笔记：[TensorFlow_learning_notes](https://github.com/cookeem/TensorFlow_learning_notes)
- 黄文坚/唐源：[《TensorFlow实战》](https://book.douban.com/subject/26974266/) 出版时间 2017-2-1
  - 源码实现：<https://github.com/terrytangyuan/tensorflow-in-practice-code>
- 掘金翻译：[TensorFlow 最新官方文档中文版 V1.10 ](https://github.com/xitu/tensorflow-docs)
- 极客学院：[TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)
- [TensorFlow 官方文档中文版](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)

## 2. 笔记

### (1) 第一部分

学习 TensorFlow 之前，先学习掌握以下内容，包括 Python 基础、Anconada 安装等等：

- [IDE之PyCharm的设置和Debug入门](./other/IDE之PyCharm的设置和Debug入门.md)
- [Python基础入门笔记（一）](./other/Python/Python基础入门笔记（一）.md)
- [Python基础入门笔记（二）](./other/Python/Python基础入门笔记（二）.md)
- [Python函数使用整理](./other/Python/Python函数使用整理.md)
- [Anaconda的介绍、安装和环境管理](./other/Anaconda的介绍、安装和环境管理.md)
- [Jupyter Notebook的介绍、安装及使用](./other/Jupyter的介绍、安装及使用.md)
- [深度学习硬件选购及tensorflow各系统下的环境搭建](./other/深度学习硬件选购及tensorflow各系统下的环境搭建.md)
- [Python常用科学计算库快速入门(NumPy、SciPy、Pandas、Matplotlib、Scikit-learn)](./other/Python常用科学计算库快速入门(NumPy、SciPy、Pandas、Matplotlib、Scikit-learn).md)
  - [numpy学习笔记](./other/科学计算库之numpy的使用.md)
  - [pandas学习笔记](./other/科学计算库之pandas的使用.md)
  - [matplotlib学习笔记](./other/科学计算库之matplotlib的使用.md)
  - [scikit-learn学习笔记](./other/科学计算库之scikit-learn的使用.md)
- scikit-learn 学习，网上资料：
  - [Sklearn Universal Machine Learning Tutorial Series | 莫烦Python](https://morvanzhou.github.io/tutorials/machine-learning/sklearn/)
  - [scikit-learn教程 -  scikit-learn 0.20.2文档](https://www.studyai.cn/tutorial/index.html)
  - [scikit-learn（sklearn） 中文文档 - ApacheCN](https://github.com/apachecn/scikit-learn-doc-zh)
- [Python图像处理笔记(含opencv-python、scikit-image等)](./other/Python图像处理笔记.md)  |  可能用到 matlab，可以学习下：[matlab学习](./other/matlab学习.md)
- [深度学习框架对比.md](./other/深度学习框架对比.md)
- ……

### (2) tensorflow学习

《深度学习框架Tensorflow学习与应用》笔记索引（其中会有补充一些内容）：

- [01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装](./Notes/01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装.md)
- [02-Tensorflow的基础使用，包括对图(graphs),会话(session),张量(tensor),变量(Variable)的一些解释和操作](./Notes/02-Tensorflow的基础使用，包括对图\(graphs\),会话\(session\),张量\(tensor\),变量\(Variable\)的一些解释和操作.md)
- [03-Tensorflow线性回归以及分类的简单使用](./Notes/03-Tensorflow线性回归以及分类的简单使用.md)
  - 开始以手写数字识别 MNIST 例子来讲解，关于 MNIST  的内容还可以看看该 README 下面的
- [04-softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍](./Notes/04-softmax，交叉熵\(cross-entropy\)，dropout以及Tensorflow中各种优化器的介绍.md)
  - softmax、损失函数、dropout
  - tensorflow 中各种优化器
  - 在（三）节开始的代码`4-1交叉熵.py`，发现 tf.nn.softmax_cross_entropy_with_logits 用法的小问题，[详见-传送](./Notes/tf.nn.softmax_cross_entropy_with_logits的用法问题.md)
- [05-使用Tensorboard进行结构可视化，以及网络运算过程可视化](./Notes/05-使用Tensorboard进行结构可视化，以及网络运算过程可视化.md)
  - 用例子演示如何使结构的可视化
  - 参数细节的可视化，绘制各个参数变化情况
  - 补充内容：可视化工具 TensorBoard 更多使用和细节
- [06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题](./Notes/06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题.md)
  - 卷积神经网络 CNN（包括局部感受野、权值共享、卷积、二维池化、多通道池化等）
  - 补充内容：参数数量的计算（以 LeNet-5 为例子）
  - 补充内容：TensorFlow 中的 Padding 到底是怎样的？ 
  - 补充内容：TensorFlow 中的卷积和池化 API 详解
  - 补充内容：TensorFlow 中的 Summary 的用法
- [07-递归神经网络LSTM的讲解，以及LSTM网络的使用](./Notes/07-递归神经网络LSTM的讲解，以及LSTM网络的使用.md)
- [08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别](./Notes/08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别.md)
  - 保存模型、加载模型
  - 使用 Inception-v3 网络模型进行图像识别
  - 补充内容：加载预训练模型和保存模型以及 fine-tuning
  - 补充内容：迁移学习
- [09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别](./Notes/09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别.md)
  - TensorFlow 的 GPU 版本安装
  - 使用 inception-v3 模型进行训练预测
  - 使用 tensorflow 已经训练好的模型进行微调
  - 制作 `.tfrecord` 存储文件
- [10-使用Tensorflow进行验证码识别](./Notes/10-使用Tensorflow进行验证码识别.md)
- [11-Tensorflow在NLP中的使用(一)](./Notes/11-Tensorflow在NLP中的使用\(一\).md)
- [12-Tensorflow在NLP中的使用(二)](./Notes/12-Tensorflow在NLP中的使用\(二\).md)

补充的内容：

- 对 TensorFlow 的再次理解和总结：[TensorFlow的理解和总结](./other/[转]TensorFlow的理解和总结.md)
- 对 TensorFlow 的 API 使用记录下来，方便查阅：:mag_right: ​[TensorFlow的API详解和记录](./other/[整理]TensorFlow的API详解和记录.md)  这里没记录的和没记全的内容可以下面【:name_badge: <a href="#bowen">网上博文</a>】找找看！！！
- TensorFlow 使用指定的 GPU 以及显存分析：[tensorflow中使用指定的GPU及显存分析](./other/tensorflow中使用指定的GPU及显存分析.md)

<a name="bowen"></a>

:name_badge: 网上博文：

- 关于**损失函数(或代价函数)**：[Tensorflow基础知识---损失函数详解](https://sthsf.github.io/wiki/Algorithm/DeepLearning/Tensorflow%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/Tensorflow%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86---%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E8%AF%A6%E8%A7%A3.html)  | [深度学习中常用的损失函数有哪些（覆盖分类，回归，风格化，GAN等任务）？](<https://zhuanlan.zhihu.com/p/60302475>) [荐]  
- 自定义损失函数：[tensorflow内置的四个损失函数](https://blog.csdn.net/limiyudianzi/article/details/80693695) [荐]  | [自定义损失函数](https://blog.csdn.net/limiyudianzi/article/details/80697711)  |  [二分类、多分类与多标签问题的区别,对应损失函数的选择,你知道吗？ - 掘金](<https://juejin.im/post/5b38971be51d4558b10aad26>)  [荐]
- 关于 **tensorflow 中优化器**：[个人笔记-优化器参数详解（learning rate、weight decay、momentum、滑动平均等）](./other/tensorflow优化器参数详解.md)  [荐]  |  [第三章（1.5）关于tensorflow优化器 optimizer 的选择](https://blog.csdn.net/lzc4869/article/details/78355132) [荐] | [深度学习——优化器算法Optimizer详解（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）](https://www.cnblogs.com/guoyaohua/p/8542554.html)  [荐]
- CNN网络架构演进：[一文总览CNN网络架构演进：从LeNet到DenseNet](https://mp.weixin.qq.com/s/aJZ3T8EVaGDGfqxIs2av6A) [荐]
- 学习使用 TensorBoard 可视化：[详解 TensorBoard－如何调参](https://blog.csdn.net/aliceyangxi1987/article/details/71716596) | [[干货|实践] TensorBoard可视化 - 知乎](https://zhuanlan.zhihu.com/p/33178205)
- tensorflow 模型的保存和读取：[TensorFlow学习笔记（8）--网络模型的保存和读取](https://blog.csdn.net/lwplwf/article/details/62419087)
- ……

### (3) 其他

- [Batch Normalization学习笔记及其实现 - 知乎](<https://zhuanlan.zhihu.com/p/26138673>)



## 3. MNIST

- [MNIST数据集二进制格式转换为图片](./other/MNIST/MNIST数据集二进制格式转换为图片.md)
- [手写数字识别MNIST讲解](./other/MNIST/手写数字识别MNIST讲解.md)

## 4. TF快速入门总结

参考「机器之心」编译文章：

- [令人困惑的TensorFlow！](https://zhuanlan.zhihu.com/p/38812133)
- [令人困惑的 TensorFlow！(II)](https://zhuanlan.zhihu.com/p/46008208)





<div align="right">
        <a href="#top">回到顶部</a>
</div>