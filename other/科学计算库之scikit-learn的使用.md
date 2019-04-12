# scikit-learn 学习

## 1. sklearn.preprocessing.LabelEncoder

sklearn.preprocessing.LabelEncoder()：标准化标签，将标签值统一转换成range(标签值个数-1)范围内

以数字标签为例：

``` xml
In [1]: from sklearn import preprocessing
   ...: le = preprocessing.LabelEncoder()
   ...: le.fit([1,2,2,6,3])
   ...:
Out[1]: LabelEncoder()
```



参考：[sklearn.preprocessing.LabelEncoder](<https://blog.csdn.net/kancy110/article/details/75043202>)