# TensorFlow学习记录

[![license](https://img.shields.io/badge/license-Attribution--NonCommercial%204.0%20-brightgreen.svg)](https://github.com/doocs/advanced-java/blob/master/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](http://makeapullrequest.com)

## (1) 前言

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

- 油管视频：[TF Girls 修炼指南](https://www.youtube.com/watch?v=TrWqRMJZU8A&list=PLwY2GJhAPWRcZxxVFpNhhfivuW0kX15yG&index=2) 、或 B 站观看： [TF Girls 修炼指南](https://space.bilibili.com/16696495/#/channel/detail?cid=1588) 
- 油管视频：51CTO视频 [深度学习框架-Tensorflow案例实战视频课程](https://www.youtube.com/watch?v=-pYU4ub7g0c&list=PL8LR_PrSuIRhpEYA3sJ-J5hYGYUSwZwdS)、或 B 站观看：[深度学习框架-Tensorflow案例实战视频课程](https://www.bilibili.com/video/av29663946/?p=1)

(3) 相关资料：

- 郑泽宇/顾思宇：[《Tensorflow：实战Google深度学习框架》](https://book.douban.com/subject/26976457/) 出版时间 2017-2-10
  - 官方维护的书中的 TensorFlow 不同版本的示例程序仓库：<https://github.com/caicloud/tensorflow-tutorial>；
  - GitHub 有人写了笔记：[TensorFlow_learning_notes](https://github.com/cookeem/TensorFlow_learning_notes)
- 黄文坚/唐源：[《TensorFlow实战》](https://book.douban.com/subject/26974266/) 出版时间 2017-2-1
  - 源码实现：<https://github.com/terrytangyuan/tensorflow-in-practice-code>
- 掘金翻译：[TensorFlow 最新官方文档中文版 V1.10 ](https://github.com/xitu/tensorflow-docs)
- 极客学院：[TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)
- [TensorFlow 官方文档中文版](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)

## (2) 笔记索引

《深度学习框架Tensorflow学习与应用》笔记：

- [01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装](./Notes/01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装.md)
- [02-Tensorflow的基础使用，包括对图(graphs),会话(session),张量(tensor),变量(Variable)的一些解释和操作](./Notes/02-Tensorflow的基础使用，包括对图\(graphs\),会话\(session\),张量\(tensor\),变量\(Variable\)的一些解释和操作.md)
- [03-Tensorflow线性回归以及分类的简单使用](./Notes/03-Tensorflow线性回归以及分类的简单使用.md)
- [04-softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍](./Notes/04-softmax，交叉熵\(cross-entropy\)，dropout以及Tensorflow中各种优化器的介绍.md)
- [05-使用Tensorboard进行结构可视化，以及网络运算过程可视化](./Notes/05-使用Tensorboard进行结构可视化，以及网络运算过程可视化.md)
- [06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题](./Notes/06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题.md)
- [07-递归神经网络LSTM的讲解，以及LSTM网络的使用](./Notes/07-递归神经网络LSTM的讲解，以及LSTM网络的使用.md)
- [08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别](./Notes/08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别.md)
- [09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别](./Notes/09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别.md)
- [10-使用Tensorflow进行验证码识别](./Notes/10-使用Tensorflow进行验证码识别.md)
- [11-Tensorflow在NLP中的使用(一)](./Notes/11-Tensorflow在NLP中的使用\(一\).md)
- [12-Tensorflow在NLP中的使用(二)](./Notes/12-Tensorflow在NLP中的使用\(二\).md)

对 TensorFlow 的再次理解和总结：[TensorFlow的理解和总结](./other/[转]TensorFlow的理解和总结.md)

对 TensorFlow 的 API 使用记录下来，方便查阅​ :mag_right:：[TensorFlow的API使用查阅](./other/[总结]TensorFlow的API使用查阅.md) 

## (3) 笔记说明

(1) 在【[03-Tensorflow线性回归以及分类的简单使用](./Notes/03-Tensorflow线性回归以及分类的简单使用.md)】中（二）节开始以手写数字识别 MNIST 例子来讲解，关于 MNIST  的内容还可以看看该 README 下面的。

(2) 发现了问题：在【[04-softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍](./Notes/04-softmax，交叉熵\(cross-entropy\)，dropout以及Tensorflow中各种优化器的介绍.md)】中（三）节开始的代码`4-1交叉熵.py`，可以注意到如下代码：

``` python
# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 这里使用对数释然代价函数tf.nn.softmax_cross_entropy_with_logits()来表示跟softmax搭配使用的交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
```

我觉得这里是有问题的，问题在哪呢？先看[【TensorFlow】tf.nn.softmax_cross_entropy_with_logits的用法](https://blog.csdn.net/zj360202/article/details/78582895)该文，可以了解到 `tf.nn.softmax_cross_entropy_with_logits` 函数的 logits 参数传入的是未经过 softmax 的 label 值。

``` python
import tensorflow as tf  

#our NN's output  
logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
```

``` python
#step1:do softmax  
y=tf.nn.softmax(logits)  
#true label  
y_=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])  
#step2:do cross_entropy  
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  
```

两步可以用这一步代替：

``` python
#do cross_entropy just one step  
cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y_))#dont forget tf.reduce_sum()!! 
```

但`prediction = tf.nn.softmax(tf.matmul(x, W) + b)`这里的 prediction 已经经历了一次 softmax，然后又经过了 tf.nn.softmax_cross_entropy_with_logits 函数，这相当于经过两个 softmax 了（虽然大的值的概率值还是越大，这点上倒是没影响。）

关于 softmax_cross_entropy_with_logits，这篇文章也有提到【[TensorFlow入门](https://statusrank.xyz/articles/31693f5f.html)】：

> 这个函数内部自动计算 softmax 函数，然后在计算代价损失,所以logits必须是未经 softmax 处理过的函数，否则会造成错误。

注1：好像后面的笔记中程序代码都是这样的，我觉得可能视频讲解老师没注意到这点问题。另外，在该文 [06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题](./Notes/06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题.md) 的笔记中，我也记录了该问题。

注2：对 softmax、softmax loss、cross entropy 不了解，可以看下网上该文 [卷积神经网络系列之softmax，softmax loss和cross entropy的讲解](https://blog.csdn.net/u014380165/article/details/77284921)，讲解地非常清楚。【荐】

另外，关于在 TensorFlow 中有哪些损失函数的实现呢？看看该文：[tensorflow API:tf.nn.softmax_cross_entropy_with_logits()等各种损失函数](https://blog.csdn.net/NockinOnHeavensDoor/article/details/80139685)

(3) 在【 [05-使用Tensorboard进行结构可视化，以及网络运算过程可视化](./Notes/05-使用Tensorboard进行结构可视化，以及网络运算过程可视化.md)】该文可以学习到：

- 用例子演示如何使结构的可视化
- 参数细节的可视化，绘制各个参数变化情况
- 补充内容：可视化工具 TensorBoard 更多使用和细节★（这部分会不断补充和更新的…）

(4) 在【[06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题](./Notes/06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题.md)】该文可以学习到：

- 卷积神经网络 CNN（包括局部感受野、权值共享、卷积、池化）
- 补充内容：参数数量的计算（以 LeNet-5 为例子）
- 补充内容：TensorFlow 中的 Padding 到底是怎样的？★ （这个认真看下~）
- 补充内容：TensorFlow 中的 Summary 的用法



## (4) 深度学习版「Hello World」MNIST学习

- [MNIST数据集二进制格式转换为图片](./other/MNIST/MNIST数据集二进制格式转换为图片.md)
- [手写数字识别MNIST讲解](./other/MNIST/手写数字识别MNIST讲解.md)


## (5) TensorFlow快速入门总结

参考「机器之心」编译文章：

- [令人困惑的TensorFlow！](https://zhuanlan.zhihu.com/p/38812133)
- [令人困惑的 TensorFlow！(II)](https://zhuanlan.zhihu.com/p/46008208)

## 关于我

欢迎关注公众号：一个程序员的随想

![](./wechat.jpg)

<div align="right">
        <a href="#Tensorflow学习">回到顶部</a>
</div>