tensor张量、tensor的属性、tensor数据和numpy数据的转化：https://blog.csdn.net/promisejia/article/details/80806916



### tensor的属性

- **name属性**：**name是一个Tensor的唯一标识符**.
- **shape属性**：描述**维数信息**
- **dtype属性**： tensor的数据类型，TensorFlow会对所有参与计算的Tensor进行类型检查，当发现**类型不匹配时会报错**，例如下面程序中b改为`dtype= tf.float64`,则会报错。 



``` python
import tensorflow as tf
a = tf.constant([1.0,2.0],name='A'，dtype= tf.float32) 
b = tf.constant([2.0,3.0],name='A'，dtype= tf.float32) #指定同一个name，tf会自动加_k加以区分
r = tf.add(a,b)#没指定name，默认为操作名

print(a)
print(b) #打印tensor
print(r)

out:
Tensor("A:0", shape=(2,), dtype=float32)
Tensor("A_1:0", shape=(2,), dtype=float32)
Tensor("Add:0", shape=(2,), dtype=float32)
```





### numpy 数据和tensor数据转换

tensor其实是一种可以表示多维数组的class类，和 
numpy可以互相转化。  函数形式：`tf.convert_to_tensor(arr)`

```python
import tensorflow as tf
import numpy as np

arr = np.ones([2,3])
print(type(arr))

tensor = tf.convert_to_tensor(arr,name='x')  # ndarrray ---->tensor
print(type(tensor))

with tf.Session() as sess:
    print(sess.run(tensor))
```