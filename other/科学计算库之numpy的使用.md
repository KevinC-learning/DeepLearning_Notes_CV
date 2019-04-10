

# numpy 学习

## random 函数

参考：

- [为什么你用不好Numpy的random函数？](https://www.cnblogs.com/lemonbit/p/6864179.html)

### 1. numpy.random.rand()

numpy.random.rand(d0,d1,…,dn)

- rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
- dn表格每个维度
- 返回值为指定维度的array

np.random.rand(4,2)





### 2. numpy.random.randn()



### 3. numpy.random.randint()



### 4. 生成[0,1)之间的浮点数



### 5. numpy.random.choice()



### 6. numpy.random.seed()

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









