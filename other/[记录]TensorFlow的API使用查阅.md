

###【TensorFlow】tensorflow中的参数初始化方法

[tensorflow中的参数初始化方法](https://blog.csdn.net/dcrmg/article/details/80034075)

- 初始化为常量，tf.zeros_initializer() 和 tf.ones_initializer() 类，分别用来初始化全 0 和全 1 的tensor对象。

- 初始化为正太分布

  - tf中使用 tf.random_normal_initializer() 类来生成一组符合**标准正太分布**的tensor。
  - tf中使用 tf.truncated_normal_initializer() 类来生成一组符合**截断正太分布**的tensor。

- 初始化为均匀分布，tf中使用 tf.random_uniform_initializer 类来生成一组符合均匀分布的 tensor。

  > 从输出可以看到，均匀分布生成的随机数并不是从小到大或者从大到小均匀分布的，这里均匀分布的意义是每次从一组服从均匀分布的数里边随机抽取一个数。
  >
  > tf中另一个生成均匀分布的类是 tf.uniform_unit_scaling_initializer()，同样都是生成均匀分布，tf.uniform_unit_scaling_initializer 跟 tf.random_uniform_initializer 不同的地方是前者不需要指定最大最小值，是通过公式计算出来的：
  >
  > ``` xml
  > max_val = math.sqrt(3 / input_size) * factor
  > min_val = -max_val
  > ```

- 初始化为变尺度正太、均匀分布，tf中tf.variance_scaling_initializer()类可以生成截断正太分布和均匀分布的tensor，增加了更多的控制参数。

- 其他初始化方式

  - tf.orthogonal_initializer() 初始化为正交矩阵的随机数，形状最少需要是二维的
  - tf.glorot_uniform_initializer() 初始化为与输入输出节点数相关的均匀分布随机数
  - tf.glorot_normal_initializer（） 初始化为与输入输出节点数相关的截断正太分布随机数


tf.truncated_normal的用法

tf.truncated_normal(shape, mean, stddev) ：shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。

> 注：关于什么是标准差请阅读该文【[标准差和方差](https://www.shuxuele.com/data/standard-deviation.html)】

例如：

``` python
import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;
 
c = tf.truncated_normal(shape=[10,10], mean=0, stddev=1)
 
with tf.Session() as sess:
	print sess.run(c)
```

参考：https://blog.csdn.net/UESTC_C2_403/article/details/72235565

### 【TensorFlow】tf.trainable_variables和tf.all_variables的对比

tf.trainable_variables 返回的是需要训练的变量列表

tf.all_variables 返回的是所有变量的列表

例如：

``` python
import tensorflow as tf;  
import numpy as np;  
import matplotlib.pyplot as plt;  
 
v = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='v')
v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')
 
global_step = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='global_step', trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99, global_step)
 
for ele1 in tf.trainable_variables():
	print ele1.name
for ele2 in tf.all_variables():
	print ele2.name
```

输出：

``` xml
v:0
v1:0

v:0
v1:0
global_step:0
```

分析：上面得到两个变量，后面的一个得到上三个变量，因为 global_step 在声明的时候**说明不是训练变量，用来关键字 trainable=False。** 





---

### 【TensorFlow】损失函数（或代价函数）

#### 前言 

在机器学习中，loss function（损失函数）也称cost function（代价函数），是用来计算预测值和真实值的差距。 
然后以loss function的最小值作为目标函数进行反向传播迭代计算模型中的参数，这个让loss function的值不断变小的过程称为优化。 

常见的损失函数：`Zero-one Loss（0-1损失）`，`Perceptron Loss（感知损失）`，`Hinge Loss（Hinge损失）`，`Log Loss（Log损失）`，`Cross Entropy（交叉熵）`，`Square Loss（平方误差）`，`Absolute Loss（绝对误差）`，`Exponential Loss（指数误差）`等

一般来说，对于分类或者回归模型进行评估时，需要**使得模型在训练数据上的损失函数值最小，即使得经验风险函数(Empirical risk)最小化**，但是如果只考虑经验风险，容易出现过拟合，因此还需要考虑模型的泛化性，一般常用的方法就是在目标函数中加上正则项，由**损失项（loss term）加上正则项（regularization term）构成结构风险（Structural risk）**，那么损失函数变为：L=∑i=1nl(yi,f(xi))+λR(w)

其中λλ为正则项超参数，常用的正则化方法包括：**L1正则和L2正则**。



#### tf.nn.softmax_cross_entropy_with_logits

`tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)`

参考：https://blog.csdn.net/zj360202/article/details/78582895

第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes

第二个参数labels：实际的标签，大小同上

具体的执行流程大概分为两步：

1. 第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，输出就是一个`num_classes`大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率），softmax的公式为：![](https://img-blog.csdn.net/20161128203449282)

2. 第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵，公式如下：![](https://img-blog.csdn.net/20161128203840317)

   其中![img](https://img-blog.csdn.net/20161128204121097)指代实际的标签中第i个的值（用mnist数据举例，如果是3，那么标签是[0，0，0，1，0，0，0，0，0，0]，除了第4个值为1，其他全为0）

   ![img](https://img-blog.csdn.net/20161128204500600)就是softmax的输出向量[Y1，Y2,Y3...]中，第i个元素的值

   显而易见，预测![img](https://img-blog.csdn.net/20161128204500600)越准确，结果的值越小（别忘了前面还有负号），最后求一个平均，得到我们想要的loss

**注意！！！**这个函数的返回值并不是一个数，**而是一个向量**，如果要求交叉熵，我们要再做一步tf.reduce_sum操作，就是对向量里面所有元素求和，最后才得到![img](https://img-blog.csdn.net/20161128213206933)，如果求loss，则要做一步`tf.reduce_mean操作`，对向量求均值！

直接看代码：

``` python
import tensorflow as tf  
  
#our NN's output  
logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
#step1:do softmax  
y=tf.nn.softmax(logits)  
#true label  
y_=tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])  
#step2:do cross_entropy  
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  
#do cross_entropy just one step  
cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y_))#dont forget tf.reduce_sum()!!  
  
with tf.Session() as sess:  
    softmax=sess.run(y)  
    c_e = sess.run(cross_entropy)  
    c_e2 = sess.run(cross_entropy2)  
    print("step1:softmax result=")  
    print(softmax)  
    print("step2:cross_entropy result=")  
    print(c_e)  
    print("Function(softmax_cross_entropy_with_logits) result=")  
    print(c_e2)  
```

输出结果是：

``` xml
step1:softmax result=
[[0.09003057 0.24472848 0.66524094]
 [0.09003057 0.24472848 0.66524094]
 [0.09003057 0.24472848 0.66524094]]
step2:cross_entro	py result=
1.222818
Function(softmax_cross_entropy_with_logits) result=
1.2228179
```



#### tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)

该函数与`tf.nn.softmax_cross_entropy_with_logits()`函数十分相似，**唯一的区别在于labels的shape，该函数的labels要求是排他性的即只有一个正确的类别，**如果`labels`的每一行不需要进行`one_hot`表示，可以使用`tf.nn.sparse_softmax_cross_entropy_with_logits()`。

> 这个函数和tf.nn.softmax_cross_entropy_with_logits函数比较明显的区别在于它的参数labels的不同，这里的参数label是非稀疏表示的，比如表示一个3分类的一个样本的标签，稀疏表示的形式为[0,0,1]这个表示这个样本为第3个分类，而非稀疏表示就表示为2（因为从0开始算，0,1,2,就能表示三类），同理[0,1,0]就表示样本属于第二个分类，而其非稀疏表示为1。tf.nn.sparse_softmax_cross_entropy_with_logits（）比tf.nn.softmax_cross_entropy_with_logits多了一步将labels稀疏化的操作。因为深度学习中，图片一般是用非稀疏的标签的，所以用tf.nn.sparse_softmax_cross_entropy_with_logits（）的频率比tf.nn.softmax_cross_entropy_with_logits高。
>
> 参考：https://blog.csdn.net/m0_37041325/article/details/77043598



#### tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)

`sigmoid_cross_entropy_with_logits`是TensorFlow最早实现的交叉熵算法。这个函数的输入是logits和labels，logits就是神经网络模型中的 W * X矩阵，注意不需要经过`sigmoid`，而`labels`的`shape`和`logits`相同，就是正确的标签值，例如这个模型一次要判断100张图是否包含10种动物，这两个输入的shape都是[100, 10]。**注释中还提到这10个分类之间是独立的、不要求是互斥，这种问题我们称为多目标（多标签）分类，例如判断图片中是否包含10种动物中的一种或几种，标签值可以包含多个1或0个1**。



#### tf.nn.weighted_cross_entropy_with_logits(logits, targets, pos_weight, name=None)

weighted_sigmoid_cross_entropy_with_logits是sigmoid_cross_entropy_with_logits的拓展版，多支持一个pos_weight参数，在传统基于sigmoid的交叉熵算法上，正样本算出的值乘以某个系数。



参考：[tensorflow API:tf.nn.softmax_cross_entropy_with_logits()等各种损失函数](https://blog.csdn.net/NockinOnHeavensDoor/article/details/80139685)



### 【TensorFlow】优化器（梯度下降）

Tensorflow 提供了下面这些种优化器：

- tf.train.GradientDescentOptimizer

- tf.train.AdadeltaOptimizer

- tf.train.AdagradOptimizer

- tf.train.AdagradDAOptimizer

- tf.train.MomentumOptimizer

- tf.train.AdamOptimizer

  > AdamOptimizer 通过使用动量（参数的移动平均数）来改善传统梯度下降，促进超参数动态调整。
  >
  > 自适应优化算法通常都会得到比SGD算法性能更差（经常是差很多）的结果，尽管自适应优化算法在训练时会表现的比较好，因此使用者在使用自适应优化算法时需要慎重考虑！（终于知道为啥CVPR的paper全都用的SGD了，而不是用理论上最diao的Adam）
  >
  > https://blog.csdn.net/weixin_38145317/article/details/79346242

- tf.train.FtrlOptimizer

- tf.train.ProximalGradientDescentOptimizer

- tf.train.ProximalAdagradOptimizer

- tf.train.RMSPropOptimizer

各种优化器对比：

- 标准梯度下降法：标准梯度下降先计算所有样本汇总误差，然后根据总误差来更新权值
- 随机梯度下降法：随机梯度下降随机抽取一个样本来计算误差，然后更新权值
- 批量梯度下降法：批量梯度下降算是一种折中的方案，从总样本中选取一个批次（比如一共有 10000 个样本，随机选取 100 个样本作为一个 batch），然后计算这个 batch 的总误差，根据总误差来更新权值。

> 标准梯度下降法：先计算所有样本汇总误差，然后根据总误差来更新全值（时间长，值更加可靠）
>
> 随机梯度下降法：随机抽取一个样本来计算误差，然后更新权值（时间短，值相对不可靠）
>
> 批量梯度下降法：从总样本中选取一个批次，然后计算这个batch的总误差，根据总误差来更新权值（折中）
>
> 作者：知乎ID-鹏、链接：https://www.zhihu.com/question/63235995/answer/452385142

更多：

- [TensorFlow学习（四）：优化器Optimizer](https://blog.csdn.net/xierhacker/article/details/53174558)
- [AI学习笔记——Tensorflow中的Optimizer(优化器)](https://www.afenxi.com/59457.html)
- [TensorFlow 学习摘要（三） 深度学习 - TensorFlow 优化器](http://blog.720ui.com/2018/tensorflow_03_dl_tensorflow_optimizer/)




### 关于tf.control_dependencies

[TensorFlow笔记——（1）理解tf.control_dependencies与control_flow_ops.with_dependencies](https://blog.csdn.net/liuweiyuxiang/article/details/79952493)

`tf.control_dependencies(self, control_inputs)`

> 通过以上的解释，我们可以知道，该函数接受的参数control_inputs，是Operation或者Tensor构成的list。返回的是一个上下文管理器，该上下文管理器用来控制在该上下文中的操作的依赖。也就是说，上下文管理器下定义的操作是依赖control_inputs中的操作的，control_dependencies用来控制control_inputs中操作执行后，才执行上下文管理器中定义的操作。

如果我们想要确保获取更新后的参数，name我们可以这样组织我们的代码。

``` python
opt = tf.train.Optimizer().minize(loss)

with tf.control_dependencies([opt]): #先执行opt
  updated_weight = tf.identity(weight)  #再执行该操作

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  sess.run(updated_weight, feed_dict={...}) # 这样每次得到的都是更新后的weight
```

可以看到以上的例子用到了tf.identity()，至于为什么要使用tf.identity()，我在下一篇博客：TensorFlow笔记——（1）理解tf.control_dependencies与control_flow_ops.with_dependencies中有详细的解释，不懂的可以移步了解。



### 关于tf.global_variables_initializer()和tf.local_variables_initializer()区别

tf.global_variables_initializer()添加节点用于初始化所有的变量(GraphKeys.VARIABLES)。返回一个初始化所有全局变量的操作（Op）。在你构建完整个模型并在会话中加载模型后，运行这个节点。

能够将所有的变量一步到位的初始化，非常的方便。通过feed_dict, 你也可以将指定的列表传递给它，只初始化列表中的变量。

``` python
sess.run(tf.global_variables_initializer(), 
feed_dict={
        learning_rate_dis: learning_rate_val_dis,
        adam_beta1_d_tf: adam_beta1_d,
        learning_rate_proj: learning_rate_val_proj,
        lambda_ratio_tf: lambda_ratio,
        lambda_l2_tf: lambda_l2,
        lambda_latent_tf: lambda_latent,
        lambda_img_tf: lambda_img,
        lambda_de_tf: lambda_de,
        adam_beta1_g_tf: adam_beta1_g,
        }) 
# learning_rate_dis为设置的变量，learning_rate_val_dis为我设置的具体的值。后续同理
```

tf.local_variables_initializer()返回一个初始化所有局部变量的操作（Op）。初始化局部变量（GraphKeys.LOCAL_VARIABLE）。GraphKeys.LOCAL_VARIABLE中的变量指的是被添加入图中，但是未被储存的变量。关于储存，请了解tf.train.Saver相关内容，在此处不详述，敬请原谅。



### tf.InteractiveSession()与tf.Session()的区别

参考：

- [TensorFlow入门：tf.InteractiveSession()与tf.Session()区别](https://blog.csdn.net/M_Z_G_Y/article/details/80416226)

  > tf.InteractiveSession() 是一种交互式的session方式，它**让自己成为了默认的session**，也就是说用户在不需要指明用哪个session运行的情况下，就可以运行起来，这就是默认的好处。这样的话就是run()和eval()函数可以不指明session啦。（重点理解下！）

- [tf.InteractiveSession()与tf.Session()](https://blog.csdn.net/qq_14839543/article/details/77822916)
- [TensorFlow（笔记）：tf.Session()和tf.InteractiveSession()的区别](https://blog.csdn.net/u010513327/article/details/81023698)

`tf.InteractiveSession()`：它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作（operations）构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用 IPython

`tf.Session()`：需要在启动 session 之前构建整个计算图，然后启动该计算图。

意思就是在我们使用`tf.InteractiveSession()`来构建会话的时候，我们可以先构建一个 session 然后再定义操作（operation），如果我们使用`tf.Session()`来构建会话我们需要在会话构建之前定义好全部的操作（operation）然后再构建会话。

tf.Session()和tf.InteractiveSession()的区别他们之间的区别就是：后者加载自身作为默认的Session。tensor.eval()和operation.run()可以直接使用。下面这三个是等价的：

``` python
sess = tf.InteractiveSession()
```

``` python
sess = tf.Session()
with sess.as_default():
```

``` python
with tf.Session() as sess:
```

如下就会报错：

``` python
import tensorflow as tf

a = tf.constant(4)
b = tf.constant(7)
c = a + b
sess = tf.Session()
print(c.eval())
```

如果这样就没问题：

``` python
a = tf.constant(4)
b = tf.constant(7)

c = a + b
# sess = tf.Session()
with tf.Session() as sess:
    print(c.eval())
```

``` python
a = tf.constant(4)
b = tf.constant(7)

c = a + b
sess = tf.InteractiveSession()
print(c.eval())
```

## 【Tensorflow】设置自动衰减的学习率

在训练神经网络的过程中，合理的设置学习率是一个非常重要的事情。对于训练一开始的时候，设置一个大的学习率，可以快速进行迭代，在训练后期，设置小的学习率有利于模型收敛和稳定性。

``` python
tf.train.exponential_decay(learing_rate, global_step, decay_steps, decay_rate, staircase=False)
```

- learning_rate：学习率
- global_step：全局的迭代次数
- decay_steps：进行一次衰减的步数
- decay_rate：衰减率
- staircase：默认为False，如果设置为True，在修改学习率的时候会进行取整

转换方程： ![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20181227114806.png)

实例：

``` python
import tensorflow as tf
import matplotlib.pyplot as plt

start_learning_rate = 0.1
decay_rate = 0.96
decay_step = 100
global_steps = 3000

_GLOBAL = tf.Variable(tf.constant(0))
S = tf.train.exponential_decay(start_learning_rate, _GLOBAL, decay_step, decay_rate, staircase=True)
NS = tf.train.exponential_decay(start_learning_rate, _GLOBAL, decay_step, decay_rate, staircase=False)

S_learning_rate = []
NS_learning_rate = []

with tf.Session() as sess:
    for i in range(global_steps):
        print(i, ' is training...')
        S_learning_rate.append(sess.run(S, feed_dict={_GLOBAL: i}))
        NS_learning_rate.append(sess.run(NS, feed_dict={_GLOBAL: i}))

plt.figure(1)
l1, = plt.plot(range(global_steps), S_learning_rate, 'r-')
l2, = plt.plot(range(global_steps), NS_learning_rate, 'b-')
plt.legend(handles=[l1, l2, ], labels=['staircase', 'no-staircase'], loc='best')
plt.show()
```

该实例表示训练过程总共迭代3000次，每经过100次，就会对学习率衰减为原来的0.96。

参考：https://blog.csdn.net/TwT520Ly/article/details/80402803

[Tensorflow实现学习率衰减](https://blog.csdn.net/u013555719/article/details/79334359)

## 【TensorFlow】 命令行参数

[[干货|实践] Tensorflow学习 - 使用flags定义命令行参数](https://zhuanlan.zhihu.com/p/33249875)

1. 利用python的argparse包
2. tensorflow自带的app.flags实现

## tf.app.run()

[tf.app.run()](https://blog.csdn.net/helei001/article/details/51859423) ：处理flag解析，然后执行main函数，那么flag解析是什么意思呢？诸如这样的：

``` python
import tensorflow as tf

flags = tf.app.flags
############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
FLAGS = tf.app.flags.FLAGS

def main(_):
    print(FLAGS.m_plus)
    print(FLAGS.m_minus)
    print(lambda_val)
if __name__ == '__main__':
    tf.app.run()  #执行main函数  
```



## 【TensorFlow】name_scope和variable scope区别

参考：[tensorflow: name_scope 和 variable_scope区别及理解](https://blog.csdn.net/u012609509/article/details/80045529)

之所以会出现这两种类型的scope，主要是后者（variable scope）为了实现tensorflow中的变量共享机制：即为了使得在代码的任何部分可以使用某一个已经创建的变量，TF引入了变量共享机制，使得可以轻松的共享变量，而不用传一个变量的引用。具体解释如下：

1、tensorflow中创建variable的2种方式：

(1) tf.Variable()：只要使用该函数，一律创建新的variable，如果出现重名，变量名后面会自动加上后缀1，2….

``` python
import tensorflow as tf

with tf.name_scope('cltdevelop):
    var_1 = tf.Variable(initial_value=[0], name='var_1')
    var_2 = tf.Variable(initial_value=[0], name='var_1')
    var_3 = tf.Variable(initial_value=[0], name='var_1')
print(var_1.name)
print(var_2.name)
print(var_3.name)
```

结果输出如下：

``` xml
cltdevelop/var_1:0
cltdevelop/var_1_1:0
cltdevelop/var_1_2:0
```

(2) tf.get_variable()：如果变量存在，则使用以前创建的变量，如果不存在，则新创建一个变量。

``` python
import tensorflow as tf
 
with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
    var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)
 
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name)        # var1:0
    print(var2.name)        # a_name_scope/var2:0
    print(var21.name)       # a_name_scope/var2_1:0
    print(var22.name)       # a_name_scope/var2_2:0
```

可以看出使用 tf.Variable() 定义的时候, 虽然 name 都一样, 但是为了不重复变量名, Tensorflow 输出的变量名并不是一样的. 所以, 本质上 var2, var21, var22 并不是一样的变量. 而另一方面, 使用tf.get_variable()定义的变量不会被tf.name_scope()当中的名字所影响.

如果想要达到重复利用变量的效果, 我们就要使用 tf.variable_scope(), 并搭配 tf.get_variable() 这种方式产生和提取变量. 不像 tf.Variable() 每次都会产生新的变量, tf.get_variable() **如果遇到了同样名字的变量时, 它会单纯的提取这个同样名字的变量(避免产生新变量).** 而在重复使用的时候, 一定要在代码中强调 scope.reuse_variables(), 否则系统将会报错, 以为你只是单纯的不小心重复使用到了一个变量. 来源：[TensorFlow之scope命名方式](https://blog.csdn.net/buddhistmonk/article/details/79769828)

2、tensorflow中的两种作用域

1. 命名域(name scope)：通过tf.name_scope()来实现；
   变量域（variable scope）：通过tf.variable_scope()来实现；可以通过设置reuse 标志以及初始化方式来影响域下的变量。 

2. 这两种作用域都会给tf.Variable()创建的变量加上词头，而tf.name_scope对tf.get_variable()创建的变量没有词头影响，代码如下：

   ``` xml
   import tensorflow as tf
   
   with tf.name_scope('cltdevelop'):
       var_1 = tf.Variable(initial_value=[0], name='var_1')
       var_2 = tf.get_variable(name='var_2', shape=[1, ])
   with tf.variable_scope('aaa'):
       var_3 = tf.Variable(initial_value=[0], name='var_3')
       var_4 = tf.get_variable(name='var_4', shape=[1, ])
   
   print(var_1.name)
   print(var_2.name)
   print(var_3.name)
   print(var_4.name)
   ```

   结果输出如下：

   ``` xml
   cltdevelop/var_1:0
   var_2:0
   aaa/var_3:0
   aaa/var_4:0
   ```

3、tensorflow中变量共享机制的实现

在tensorflow中变量共享机制是通过tf.get_variable()和tf.variable_scope()两者搭配使用来实现的。如下代码所示：

``` python
import tensorflow as tf

with tf.variable_scope('cltdevelop'):
    var_1 = tf.get_variable('var_1', shape=[1, ])
with tf.variable_scope('cltdevelop', reuse=True):
    var_2 = tf.get_variable('var_1', shape=[1,])

print(var_1.name)
print(var_2.name)
```

结果输出如下：

``` xml
cltdevelop/var_1:0
cltdevelop/var_1:0
```

【注：】当 reuse 设置为 True 或者 tf.AUTO_REUSE 时，表示这个scope下的变量是重用的或者共享的，也说明这个变量以前就已经创建好了。但如果这个变量以前没有被创建过，则在tf.variable_scope下调用tf.get_variable创建这个变量会报错。如下：

``` xml
import tensorflow as tf

with tf.variable_scope('cltdevelop', reuse=True):
    var_1 = tf.get_variable('var_1', shape=[1, ])
```

则上述代码会报错：

``` xml
ValueErrorL Variable cltdevelop/v1 doesnot exist, or was not created with tf.get_variable()
```



## 【TensorFlow】tf.where(）用法

[tenflow 入门 tf.where(）用法](https://blog.csdn.net/ustbbsy/article/details/79564828)

where(condition, x=None, y=None, name=None)的用法

condition， x, y 相同维度，condition是bool型值，True/False

返回值是对应元素，condition中元素为True的元素替换为x中的元素，为False的元素替换为y中对应元素，x只负责对应替换True的元素，y只负责对应替换False的元素，x，y各有分工，由于是替换，返回值的维度，和condition，x ， y都是相等的。

看个例子：

``` python
import tensorflow as tf
x = [[1,2,3],[4,5,6]]
y = [[7,8,9],[10,11,12]]
condition3 = [[True,False,False],
             [False,True,True]]
condition4 = [[True,False,False],
             [True,True,False]]
with tf.Session() as sess:
    print(sess.run(tf.where(condition3,x,y)))
    print(sess.run(tf.where(condition4,x,y)))  
```





