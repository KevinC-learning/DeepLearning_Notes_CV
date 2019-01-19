

[[Python]断言assert的用法](https://blog.csdn.net/humanking7/article/details/45950781)

assert的语法格式：assert expression，它的等价语句为：

``` xml
if not expression:
    raise AssertionError
```

这段代码用来检测数据类型的断言，因为 `a_str` 是 `str` 类型，所以认为它是 `int` 类型肯定会引发错误。

``` xml
>>> a_str = 'this is a string'
>>> type(a_str)
<type 'str'>
>>> assert type(a_str)== str
>>> assert type(a_str)== int

Traceback (most recent call last):
  File "<pyshell#41>", line 1, in <module>
    assert type(a_str)== int
AssertionError
```

