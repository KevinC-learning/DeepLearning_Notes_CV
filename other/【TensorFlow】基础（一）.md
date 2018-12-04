### 张量(tensor)、张量属性、张量数据和numpy数据的转化

1、张量（tensor）是 tensorflow 中的数据形式。是一种可以表示多维数组的 class 类，可以理解为多维数组。

2、张量的属性

在 tensor 类中包含以下几个属性：

- name 属性
- shape 属性
- dtype 属性

3、numpy 数据和 tensor 数据的转化

tensor 其实是一种可以表示多维数组的 class 类，和  numpy 可以互相转化。 

``` python
import tensorflow as tf
import numpy as np

arr = np.ones([2,3])
print(type(arr))

tensor = tf.convert_to_tensor(arr,name='x')  # ndarrray ---->tensor
print(type(tensor))

with tf.Session() as sess:
    print(sess.run(tensor))
```

PS：在 tensorflow 的类似 tf.constant() 的方法的参数可以是 List（列表，eg：`[2, 3, 5]`）、ndarry（多维数组， eg：`[[2, 3], [4, 5]]`）。



