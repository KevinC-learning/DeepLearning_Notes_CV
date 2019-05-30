弄清楚典型的 python 文件结构如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190530192657.png)

总结：

C++中 `main函数` 为执行的起点；Python 中首先执行最先出现的非函数定义和非类定义的没有缩进的代码，会从前到后执行。

程序中为了区分主动执行还是被调用，Python 引入了变量 `__name__`，当文件是被调用时，`__name__` 的值为模块名，当文件被执行时，`__name__` 为 `__main__`。因此，若是文件主动执行了，则最好写成跟上面的例子一样， main 之前不要有可执行代码，这样做到程序从 main（）开始，流程逻辑性强；若是文件作为模块被调用，则可以不用写 main（），从上而下顺序执行。

``` python
#test1
print ("test1")
def Fun():
    print ("Fun")
def main():
    print ("main")
    Fun()
if __name__ == '__main__':
    main()
'''
test1
main
Fun
'''
```













参考：[Python程序执行顺序](<https://blog.csdn.net/kunpengtingting/article/details/80178618>)

