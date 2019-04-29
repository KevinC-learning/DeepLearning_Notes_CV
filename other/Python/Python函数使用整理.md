## [[Python]断言assert的用法](https://blog.csdn.net/humanking7/article/details/45950781)

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



## Python 中 yied 的使用

我们先看：

``` python
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        print b 
        a, b = b, a + b 
        n = n + 1
fab(5)
```

执行以上代码，我们可以得到如下输出：

``` xml
1 
1 
2 
3 
5
```

结果没有问题，但有经验的开发者会指出，直接在 fab 函数中用 print 打印数字会导致该函数可复用性较差，因为 fab 函数返回 None，其他函数无法获得该函数生成的数列。

要提高 fab 函数的可复用性，最好不要直接打印出数列，而是返回一个 List。以下是 fab 函数改写后的第二个版本：

``` python
def fab(max): 
    n, a, b = 0, 0, 1 
    L = [] 
    while n < max: 
        L.append(b) 
        a, b = b, a + b 
        n = n + 1 
    return L
 
for n in fab(5): 
    print n
```

可以使用如下方式打印出 fab 函数返回的 List：

``` xml
1 
1 
2 
3 
5
```

改写后的 fab 函数通过返回 List 能满足复用性的要求，但是更有经验的开发者会指出，该函数在运行中占用的内存会随着参数 max 的增大而增大，如果要控制内存占用，最好不要用 List

…

使用 yield 的第四版：

``` python
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b      # 使用 yield
        # print b 
        a, b = b, a + b 
        n = n + 1
 
for n in fab(5): 
    print n
```

第四个版本的 fab 和第一版相比，仅仅把 print b 改为了 yield b，就在保持简洁性的同时获得了 iterable 的效果。

调用第四版的 fab 和第二版的 fab 完全一致：

``` xml
1 
1 
2 
3 
5
```

简单地讲，yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，调用 fab(5) 不会执行 fab 函数，而是返回一个 iterable 对象！在 for 循环执行时，每次循环都会执行 fab 函数内部的代码，执行到 yield b 时，fab 函数就返回一个迭代值，下次迭代时，代码从 yield b 的下一条语句继续执行，而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，直到再次遇到 yield。

也可以手动调用 fab(5) 的 next() 方法（因为 fab(5) 是一个 generator 对象，该对象具有 next() 方法），这样我们就可以更清楚地看到 fab 的执行流程：

执行流程：

``` xml
>>>f = fab(5) 
>>> f.next() 
1 
>>> f.next() 
1 
>>> f.next() 
2 
>>> f.next() 
3 
>>> f.next() 
5 
>>> f.next() 
Traceback (most recent call last): 
 File "<stdin>", line 1, in <module> 
StopIteration
```

当函数执行结束时，generator 自动抛出 StopIteration 异常，表示迭代完成。在 for 循环里，无需处理 StopIteration 异常，循环会正常结束。

**我们可以得出以下结论：**

一个带有 yield 的函数就是一个 generator，它和普通函数不同，生成一个 generator 看起来像函数调用，但不会执行任何函数代码，直到对其调用 next()（在 for 循环中会自动调用 next()）才开始执行。虽然执行流程仍按函数的流程执行，但每执行到一个 yield 语句就会中断，并返回一个迭代值，下次执行时从 yield 的下一个语句继续执行。看起来就好像一个函数在正常执行的过程中被 yield 中断了数次，每次中断都会通过 yield 返回当前的迭代值。

yield 的好处是显而易见的，把一个函数改写为一个 generator 就获得了迭代能力，比起用类的实例保存状态来计算下一个 next() 的值，不仅代码简洁，而且执行流程异常清晰。

——from：[Python yield 使用浅析 | 菜鸟教程](<http://www.runoob.com/w3cnote/python-yield-used-analysis.html>)