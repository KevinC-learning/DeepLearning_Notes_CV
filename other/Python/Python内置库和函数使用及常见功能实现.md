# Python 函数

## 断言 assert 的用法

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

——from：[[Python]断言assert的用法](https://blog.csdn.net/humanking7/article/details/45950781)

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



# Python实现功能

## Python 保存数据到本地文件

（1）[python保存数据到本地文件](<https://blog.csdn.net/rosefun96/article/details/78877452>)

1、保存列表为.txt文件

``` python
#1/list写入txt

ipTable = ['158.59.194.213', '18.9.14.13', '58.59.14.21']  
fileObject = open('sampleList.txt', 'w')  
for ip in ipTable:  
    fileObject.write(ip)  
    fileObject.write('\n')  
fileObject.close()  
```

2、字典保存

``` python
#2/dict写入json
import json

dictObj = {  
    'andy':{  
        'age': 23,  
        'city': 'shanghai',  
        'skill': 'python'  
    },  
    'william': {  
        'age': 33,  
        'city': 'hangzhou',  
        'skill': 'js'  
    }  
}  
  
jsObj = json.dumps(dictObj)  
  
fileObject = open('jsonFile.json', 'w')  
fileObject.write(jsObj)  
fileObject.close()  
```



## Python计算程序运行时间

方法1：

``` python
import datetime
starttime = datetime.datetime.now()
#long running
#do something other
endtime = datetime.datetime.now()
print (endtime - starttime).seconds
```

datetime.datetime.now() 获取的是当前日期，在程序执行结束之后，这个方式获得的时间值为**程序执行的时间**。

方法2：

``` python
start = time.time()
#long running
#do something other
end = time.time()
print end-start
```

time.time()获取自纪元以来的当前时间（以秒为单位）。如果系统时钟提供它们，则可能存在秒的分数。所以这个地方返回的是一个浮点型类型。这里获取的也是**程序的执行时间**。

方法3：

``` python
start = time.clock()
#long running
#do something other
end = time.clock()
print end-start
```

time.clock() 返回程序开始或第一次被调用 clock() 以来的 CPU 时间。 这具有与系统记录一样多的精度。返回的也是一个浮点类型。这里获得的是**CPU的执行时间**。 

注：程序执行时间 = cpu时间 + io时间 + 休眠或者等待时间。

**方法比较：** 

- 通过对以上方法的比较我们发现，方法二的精度比较高。方法一基本上是性能最差的。这个其实是和系统有关系的。一般我们推荐使用方法二和方法三。我的系统是Ubuntu，也就是Linux系统，方法二返回的是UTC时间。 在很多系统中time.time()的精度都是非常低的，包括windows。
- python 的标准库手册推荐在任何情况下尽量使用time.clock().但是这个函数在windows下返回的是真实时间（wall time）
- 方法一和方法二都包含了其他程序使用CPU的时间。方法三只计算了程序运行CPU的时间。
- 方法二和方法三都返回的是浮点数

参考：

- [几种Python执行时间的计算方法](<https://blog.csdn.net/wangshuang1631/article/details/54286551>)

- [计算Python的代码块或程序的运行时间](<https://blog.csdn.net/chichoxian/article/details/53108365>)

---

补充：[Python计算程序运行时间](<https://blog.csdn.net/xc_zhou/article/details/80837850>)

Python中有两个模块可以完成时间操作：`time`和`datetime`

相比较而言 datetime 更强大。

如果要获取微秒级别的时间差，可以利用以下代码：

``` python
import datetime

begin = datetime.datetime.now()
end = datetime.datetime.now()
k = end - begin

print (end-begin).days # 0 天数
print (end-begin).total_seconds() # 30.029522 精确秒数
print (end-begin).seconds # 30 秒数
print (end-begin).microseconds # 29522 毫秒数
```

亲自实践某段代码：

``` python
starttime = datetime.datetime.now()
print(starttime)
#long running
#do something other
endtime = datetime.datetime.now()
print(endtime)
print((endtime - starttime).seconds)
```

结果：

``` xml
2019-06-18 00:20:31.576806
2019-06-18 00:20:41.231358
9
```

使用：`print((endtime - starttime).total_seconds)`

``` 
2019-06-18 00:21:47.085125
2019-06-18 00:21:56.225648
9.140523
```

### time.time()、time.clock() 计算运行时间

关于 time 库中表示时间的方法，官方给出了 2 种：

1.从1970-01-01 00:00:00 UTC，开始到现在所经历的时间，以浮点数的'秒'来表示

``` xml
>>>time.time()
1517362540.347517
```

2.用结构化的时间组（year,month,day,hours,minutes,seconds....）来表示从1970-01-01 00:00:00 UTC，开始到现在所经历的时间.

``` xml
>>>time.gmtime()
time.struct_time(tm_year=2018, tm_mon=1, tm_mday=31, tm_hour=1, tm_min=37, 
tm_sec=36, tm_wday=2, tm_yday=31, tm_isdst=0)
```

time包中的功能都很实用：

- **time.clock()**返回程序运行的整个时间段中中CPU运行的时间,下面会重点介绍

- **time.sleep()**爬虫中常用，让程序暂停执行指定的秒数，如time.sleep(2)

- **time.localtime()**用结构化的时间组，表示本地时间

来用 time 计算运行时间。定义一个函数run()：

```python
def run():
    start = time.time()
    for i in range(1000):
        j = i * 2 
        for k in range(j):
            t = k
            print(t)
    end = time.time()
    print('程序执行时间: ',end - start)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190618002616.png)

可以看到，程序执行时间是5.73039174079895s。

现在，让我们用 time.clock() 来看看程序执行过程中 CPU 执行了多长时间：

```python
def run2():
    start = time.clock()
    for i in range(1000):
        j = i * 2 
        for k in range(j):
            t = k
            print(t)
    end = time.clock()
    print('CPU执行时间: ',end - start)
```

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190618002703.png)

可见，此段代码 CPU 执行时间为：5.3150249999999915。

那么问题来了，CPU 真的执行了这么长时间么？会不会有什么东西是我没考虑进去的呢？

仔细看一下，这段程序主要内容就是两个for循环，for循环执行计算的时候 CPU 肯定是在运行的，那么 print() 函数打印期间这个时间段的 CPU 执行时间有没有算进去？

带着疑问，我们进行第三次测试，此次我们去掉 print()，直接让 CPU 完成整个 for 循环的计算：

```python
def run3():
    start = time.clock()
    for i in range(1000):
        j = i * 2 
        for k in range(j):
            t = k
    end = time.clock()
    print('CPU执行时间: ',end - start)
```

结果：

```text
>>> run3()
CPU执行时间: 0.04683999999997468
```

可以看见，CPU的执行时间瞬间降低到0.04s，细想一下，其实不难理解。

因为去掉了print()，所以整个run3()函数就只剩下完整的for循环，CPU可以连续执行，（不必一遍for循环一边print()来回切换），连续执行的CPU还是很快的~

所以，这给了我一个启发，以后写代码时，要精简不必要的开销，譬如经常使用print()。。。

——from：[Python计算程序运行时间—time.time()、time.clock() - 知乎](<https://zhuanlan.zhihu.com/p/33450843>)