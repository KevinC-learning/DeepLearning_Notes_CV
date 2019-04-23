# numpy 学习

## 1. random 函数

参考：

- [为什么你用不好Numpy的random函数？](https://www.cnblogs.com/lemonbit/p/6864179.html)

### numpy.random.rand()

numpy.random.rand(d0,d1,…,dn)

- rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
- dn表格每个维度
- 返回值为指定维度的array

np.random.rand(4,2)





### numpy.random.randn()



### numpy.random.randint()



### 生成[0,1)之间的浮点数



### numpy.random.choice()



### numpy.random.seed()

- np.random.seed() 的作用：使得随机数据可预测。
- 当我们设置相同的 seed，每次生成的随机数相同。如果不设置 seed，则每次会生成不同的随机数

``` python
np.random.seed(0)
np.random.rand(5)
```

``` xml
array([ 0.5488135 ,  0.71518937,  0.60276338,  0.54488318,  0.4236548 ])
```

``` python
np.random.seed(1676)
np.random.rand(5)
```

``` xml
array([ 0.39983389,  0.29426895,  0.89541728,  0.71807369,  0.3531823 ])
```

``` python
np.random.seed(1676)
np.random.rand(5)
```

``` xml
array([ 0.39983389,  0.29426895,  0.89541728,  0.71807369,  0.3531823 ])  
```

需要注意的是，seed 值的有效次数仅为一次，因此，若要保证每次产生的随机数相同，则需要在调用随机数函数之前再次使用相同的 seed 值。下面给出相应的案例，即：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190410203603.png)

在机器学习和深度学习中，如果要保证部分参数（比如W权重参数）的随机初始化值相同，可以采用这种方式来实现。——from：<https://blog.csdn.net/zenghaitao0128/article/details/78558233>



## 2. unique 函数

a = np.unique(A)，对于一维数组或者列表，unique 函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表。

``` python

import numpy as np
A = [1, 2, 2, 5,3, 4, 3]
a = np.unique(A)
B= (1, 2, 2,5, 3, 4, 3)
b= np.unique(B)
C= ['fgfh','asd','fgfh','asdfds','wrh']
c= np.unique(C)
print(a)
print(b)
print(c)
#   输出为 [1 2 3 4 5]
# [1 2 3 4 5]
# ['asd' 'asdfds' 'fgfh' 'wrh']
```

参考：[Python中numpy库unique函数解析](<https://blog.csdn.net/yangyuwen_yang/article/details/79193770>)



## 3. numpy.expand_dims 的用法

其实感觉 expand_dims(a, axis) 就是在 axis 的那一个轴上把数据加上去，这个数据在 axis 这个轴的 0 位置。 

例如原本为一维的 2 个数据，axis=0，则 shape 变为(1,2)，axis=1 则 shape 变为 (2,1) ；再例如 原本为 (2,3)，axis=0，则 shape 变为(1,2,3)，axis=1 则 shape 变为(2,1,3)

参考：[5 python numpy.expand_dims的用法](<https://blog.csdn.net/qq_16949707/article/details/53418912>)



