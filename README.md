# TensorFlow学习

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

—— 说明：实际第 5 周讲的是 Tensorborad 结构可视化，第 6 周讲的是 CNN。修正：下载链接里的文件夹顺序，我已修正。

1）在线观看：

- YouTube：[tensorflow教程（十课）](https://www.youtube.com/watch?v=eAtGqz8ytOI&list=PLjSwXXbVlK6IHzhLOMpwHHLjYmINRstrk&index=2&t=0s)
- 或 B 站：[《深度学习框架TensorFlow学习与应用》](https://www.bilibili.com/video/av20542427/)

2）下载：

- 《深度学习框架Tensorflow学习与应用》（含视频+代码+课件，视频总时长：13小时31分钟）

  > 链接: https://pan.baidu.com/s/16OINOrFiRXbqmqOFjCFzLQ 密码: 1a8j

- 《深度学习框架Tensorflow学习与应用[只有videos-720p]》（该份资料只有视频文件）

  > 链接: https://pan.baidu.com/s/1oQLgWFEBsVrcKJN4swEdzg 密码: i3e2

*其他学习视频（觉得有必要可以去看看）：* 

- 油管视频：[TF Girls 修炼指南](https://www.youtube.com/watch?v=TrWqRMJZU8A&list=PLwY2GJhAPWRcZxxVFpNhhfivuW0kX15yG&index=2) 、或 B 站观看： [TF Girls 修炼指南](https://space.bilibili.com/16696495/#/channel/detail?cid=1588) 
- 油管视频：51CTO视频 [深度学习框架-Tensorflow案例实战视频课程](https://www.youtube.com/watch?v=-pYU4ub7g0c&list=PL8LR_PrSuIRhpEYA3sJ-J5hYGYUSwZwdS)、或 B 站观看：[深度学习框架-Tensorflow案例实战视频课程](https://www.bilibili.com/video/av29663946/?p=1)

*相关资料：*

- 郑泽宇/顾思宇：[《Tensorflow：实战Google深度学习框架》](https://book.douban.com/subject/26976457/) 出版时间 2017-2-10
  - 官方维护的书中的 TensorFlow 不同版本的示例程序仓库：<https://github.com/caicloud/tensorflow-tutorial>；
  - GitHub 有人写了笔记：[TensorFlow_learning_notes](https://github.com/cookeem/TensorFlow_learning_notes)
- 黄文坚/唐源：[《TensorFlow实战》](https://book.douban.com/subject/26974266/) 出版时间 2017-2-1
  - 源码实现：<https://github.com/terrytangyuan/tensorflow-in-practice-code>
- 掘金翻译：[TensorFlow 最新官方文档中文版 V1.10 ](https://github.com/xitu/tensorflow-docs)
- 极客学院：[TensorFlow 官方文档中文版](http://wiki.jikexueyuan.com/project/tensorflow-zh/)
- [TensorFlow 官方文档中文版](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)

## 笔记索引

《深度学习框架Tensorflow学习与应用》笔记：

- [01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装](/Notes/01-Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装.md)
- [02-Tensorflow的基础使用，包括对图(graphs),会话(session),张量(tensor),变量(Variable)的一些解释和操作](/Notes/02-Tensorflow的基础使用，包括对图\(graphs\),会话\(session\),张量\(tensor\),变量\(Variable\)的一些解释和操作.md)
- [03-Tensorflow线性回归以及分类的简单使用](/Notes/03-Tensorflow线性回归以及分类的简单使用.md)
- [04-softmax，交叉熵(cross-entropy)，dropout以及Tensorflow中各种优化器的介绍](/Notes/04-softmax，交叉熵\(cross-entropy\)，dropout以及Tensorflow中各种优化器的介绍.md)
- [05-使用Tensorboard进行结构可视化，以及网络运算过程可视化](/Notes/05-使用Tensorboard进行结构可视化，以及网络运算过程可视化.md)
- [06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题-](/Notes/06-卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题.md)
- [07-递归神经网络LSTM的讲解，以及LSTM网络的使用](/Notes/07-递归神经网络LSTM的讲解，以及LSTM网络的使用.md)
- [08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别](/Notes/08-保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别.md)
- [09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别](/Notes/09-Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别.md)
- [10-使用Tensorflow进行验证码识别](/Notes/10-使用Tensorflow进行验证码识别.md)
- [11-Tensorflow在NLP中的使用(一)](/Notes/11-Tensorflow在NLP中的使用\(一\).md)
- [12-Tensorflow在NLP中的使用(二)](/Notes/12-Tensorflow在NLP中的使用\(二\).md)

----

个人补充内容：

①关于手写数字识别 MNIST

- [MNIST数据集二进制格式转换为图片](other/MNIST数据集二进制格式转换为图片.md)
- [CNN实现MNIST手写数字识别](https://blog.csdn.net/lijiecao0226/article/details/78379110)

  > 通过该文重点体会下网络结构卷积、池化过程，feature map 以及深度的变化。
  >
  > 如果不清楚，可以看下该文【[从AlexNet理解卷积神经网络的一般结构](https://blog.csdn.net/chaipp0607/article/details/72847422)】，现摘入部分内容：
  >
  > CNN 中的卷积层操作与图像处理中的卷积是一样的，都是一个卷积核对图像做自上而下，自左而右的加权和操作，不同指出在于，在传统图像处理中，我们人为指定卷积核，比如 Soble，我们可以提取出来图像的水平边缘和垂直边缘特征。而在 CNN 中，卷积核的尺寸是人为指定的，但是卷积核内的数全部都是需要不断学习得到的。比如一个卷积核的尺寸为`3×3×3`，分别是宽，高和厚度，那么这一个卷积核中的参数有 27 个。在这里需要说明一点：
  >
  > - 卷积核的厚度=被卷积的图像的通道数
  > - 卷积核的个数=卷积操作后输出的通道数
  >
  > 这两个等式关系在理解卷积层中是非常重要的！！举一个例子，输入图像尺寸`5×5×3`（宽/高/通道数）,卷积核尺寸：`3×3×3`（宽/高/厚度），步长：1，边界填充：0，卷积核数量：1。用这样的一个卷积核去卷积图像中某一个位置后，是将**该位置上宽 3，高 3，通道 3 上 27 个像素值分别乘以卷积核上 27 个对应位置的参数**，得到一个数，依次滑动，得到卷积后的图像，**这个图像的通道数为 1**（与卷积核个数相同），图像的高宽尺寸如下公式：![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105201546.png)所以，卷积后的图像尺寸为：`3×3×1`（宽/高/通道数）

- [TensorFlow入门1-CNN网络及MNIST例子讲解](https://blog.csdn.net/zgzczzw/article/details/79897956)

  > 该文讲解的挺全面的，值得一看。现摘入其中部分内容：
  >
  > 全连接层，顾名思义，指的是这一层的每个节点都跟上一层的全部节点有连接。
  >
  > ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105203937.png)
  >
  > 比如图中全连接层的节点 a1 与上一层的所有节点 x1、x2、x3 有联系，a2 也与上一层的所有节点有联系，就这样，少一个也不行。因此解释全连接层的时候，图就是上面这样一个网状结构。
  >
  > 假设 x1，x2，x3 是全连接层的输入值，那么在全连接层的 a1 可以表示为以下的形式。
  >
  > ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105204240.png)
  >
  > 从这个公式可以看出，a1 与 x1，x2，x3 都有联系，只是输入节点的权重值不同。同理，a2 和 a3 也可以表示为以下的形式。
  >
  > ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105204304.png)
  >
  > 由此就可以从数学层面去理解什么叫全连接层，但是这个全连接层在实际应用中表示得意义是什么呢，简单来说，它可以作为一个分类网络，还是上面那个图，输入层是 3 个值，全连接层是 3 个节点，就代表这个全连接层可以把上一层的特征值分类为三种特征。当然，输入层和全连接层的节点数并不一定相同，比如下面这个结构。
  >
  > ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105204341.png)
  >
  > 最后的全连接层是 10 个节点，全连接层的上一层是 15 个特征值，这样就把 15 个特征值归类为 10 种特征。举例来说，比如输出的 15 个特征标记位 t1-15，而 t1，t3，t5 三种特征可以认为都属于某一种输出 o1。而 t2，t6 两种特征可以认为属于输出 o2，以此类推，就可以把 15 个特征分类为 10 个输出。
  >
  > 前面提到的 MNIST 问题中，全连接层有两层，第一层是 1024 个节点，第二层是 10 个节点。一般情况下，最靠近用户的输出层就是用户所期待的结果类别数，MNIST 问题中，用户期待的输出是 0-9 一共 10 个数字也就是 10 种类别，所以第二层全连接层是 10 个节点。再往前看，卷积层在每一个像素点上提取 64 个特征值，所以整个图片可以有`高x宽`x64个特征值，也就是 `7x7x64` 个特征，第一个全连接层是 1024 个节点，表示希望这个网络将这 7*7*64 个特征归类为 1024 个分类。至于为什么第一个连接层是 1024 个节点，是一个经验值的问题，这个节点的个数是可以调整的，后面会讲到全连接层对模型体积的影响，也会再说到这个经验值的问题。那么全连接层是如何分类的，一起来看一下。
  >
  > （剩下内容略……）
  >
  > ---
  >
  > 个人补充：关于 softmax 函数。
  >
  > 我们知道 MNIST 的结果是 0-9，我们的模型可能推测出一张图片是数字 9 的概率是 80%，是数字 8 的概率是 10%，然后其他数字的概率更小，总体概率加起来等于 1。这是一个使用 softmax 回归模型的经典案例。softmax 模型可以用来给不同的对象分配概率。![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105205325.png)
  >
  > 比如输出结果为[1, 5, 3]：
  >
  > ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181105205403.png)
  >
  > softmax 多用于多分类神经网络输出。

- [Tensorflow 实现 MNIST 手写数字识别](https://blog.csdn.net/u010858605/article/details/69830657)

  > 通过该文重点体会下神经网络最后是如何做到数字分类的。

---

## TensorFlow快速入门总结

参考「机器之心」编译文章：

- [令人困惑的TensorFlow！](https://zhuanlan.zhihu.com/p/38812133)
- [令人困惑的 TensorFlow！(II)](https://zhuanlan.zhihu.com/p/46008208)

