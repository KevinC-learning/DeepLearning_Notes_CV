基于 python 脚本语言开发的数字图片处理包，比如 PIL、Pillow、opencv、scikit-image 等。

- PIL 和 Pillow 只提供最基础的数字图像处理，功能有限；

- opencv 实际上是一个 c++ 库，只是提供了 python 接口，更新速度非常慢。
- scikit-image 是基于 scipy 的一款图像处理包，它将图片作为 numpy 数组进行处理，正好与 matlab 一样，因此，我们最终选择 scikit-image 进行数字图像处理。



学习：[python skimage图像处理(一) - 简书](<https://www.jianshu.com/p/f2e88197e81d>)



---

# 1. opencv-python 的使用

## opencv-python安装

**（1）Windows 下的安装**

opencv 依赖 numpy，先安装好 numpy。

方法一：直接命令法

试试 pip 或 conda 命令安装 `pip install opencv-python`

方法二：下载 whl 文件安装

到官网 <https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv>，下载相应 Python 版本的 OpenCV 的 whl 文件。比如下载的是 opencv_python‑3.4.1‑cp36‑cp36m‑win_amd64.whl，则打开该 whl 文件所在目录，CMD 进入到该目录，使用命令安装即可：

``` 
pip install opencv_python‑3.4.1‑cp36‑cp36m‑win_amd64.whl
```

测试是否安装成功：

``` python
import cv2
```

运行是否报错。

注意：本人在安装 opencv-python 出现了问题，后来换了其他版本的 opencv 解决了，所以怀疑 Python 版本和 opencv-python 版本需要对应。

本人 Python 版本：3.6.4  opencv-python 版本：3.4.1.15

---



## opencv-python 图像处理

### 图像处理代码随记

（1）设置 500x500x3 图像 的 100x100 区域为蓝色：

``` python
import cv2
import numpy as np

ann_img = np.ones((500,500,3)).astype('uint8')
print(ann_img.shape)
ann_img[:100, :100, 0] = 255 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
# print(ann_img)

cv2.imshow("Image", ann_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

`ann_img[:100, :100, 0]` 表示：第一维度的 0~99 索引位置、第二维度的 0~99 索引位置，第三维度的索引为 1 的位置，这些位置的值改为 255，可以看出，分别对应高、宽、通道，高和宽的 0~99 位置的 100 个像素，以及通道中的第一个通道的值都改了为 255，所以变为了蓝色。

（2）

``` python
import cv2
import numpy as np

img = cv2.imread("./haha.jpg", cv2.IMREAD_COLOR)
print(img.shape)
print(img)
emptyImage = np.zeros(img.shape, np.uint8)
print(emptyImage)
emptyImage2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("EmptyImage", emptyImage)
cv2.imshow("Image", img)
cv2.imshow("EmptyImage2", emptyImage2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### RGB 转为灰度图像

RGB 彩色图像中，一种彩色由R（红色），G（绿色），B（蓝色）三原色按比例混合而成。

图像的基本单元是一个像素，就像一个巨幅电子广告屏上远处看是衣服图像，走近你会看到一个一个的方格，这个方格的颜色是一种，从远处看，觉察不到这个方格的存在。

一个像素需要 3 块表示，分别代表 R，G，B，如果 8 为表示一个颜色，就由 0-255 区分不同亮度的某种原色。

实际中数都是二进制形式的，并且未必按照 R，G，B 顺序，比如[OpenCV](http://lib.csdn.net/base/opencv)是按照 B,G,R 顺序将三个色值保存在 3 个连续的字节里。

**灰度图像**是用不同饱和度的黑色来表示每个图像点，比如用8位 0-255数字表示“灰色”程度，每个像素点只需要一个灰度值，8位即可，这样一个 3X3 的灰度图，只需要9个byte就能保存

RGB值和灰度的转换，实际上是人眼对于彩色的感觉到亮度感觉的转换，这是一个心理学问题，有一个公式：

**Grey = 0.299\*R + 0.587\*G + 0.114\*B**

根据这个公式，依次读取每个像素点的 R，G，B 值，进行计算灰度值（转换为整型数），将灰度值赋值给新图像的相应位置，所有像素点遍历一遍后完成转换。

——from：[RGB图像转为灰度图](https://blog.csdn.net/u010312937/article/details/71305714)

### 理解赋值生成的图像

先看这样的代码：

``` python
import cv2
import numpy as np

ann_img = np.ones((4,3,3)).astype('uint8')
print(ann_img.shape)
ann_img[:2, :2, 0] = 255
print(ann_img)
```

结果：

``` xml
(4, 3, 3)
[[[255   1   1]
  [255   1   1]
  [  1   1   1]]

 [[255   1   1]
  [255   1   1]
  [  1   1   1]]

 [[  1   1   1]
  [  1   1   1]
  [  1   1   1]]

 [[  1   1   1]
  [  1   1   1]
  [  1   1   1]]]
```

我们来这样理解下，图像高宽的像素个数分别为 4 和 3，3 个通道。我们可以把其想象成三个矩阵叠加（我暂且称它为「叠加矩阵」）。

我们再来看打印的数据值，那么这个表示叠加矩阵的第一行的像素值，

``` xml
[[255   1   1]
  [255   1   1]
  [  1   1   1]]
```

下面这个表示叠加矩阵的第二行的像素值，

``` xml
[[255   1   1]
  [255   1   1]
  [  1   1   1]]
```

依次内推。总共 4 行（正好对应高度 4）。然后再回来看叠加矩阵的第一个 `[ ]` 的像素值，其中的这个

``` xml
[255   1   1]
```

255 表示第一行第一列的第一通道的像素值，中间的 1 表示第一行第一列的第二通道的像素值，最后的 1 表示第一行第一列的第三通道的像素值。

接下来可以来看叠加矩阵的第二行像素值，第三行像素值。。。依次内推。理解起来是一样的。

### label图像

先看代码：

``` python
import cv2
import numpy as np

ann_img = np.ones((4,3)).astype('uint8')
print(ann_img.shape)
ann_img[:2, :2] = 255
print(ann_img)

print('\n---------------\n')

print(ann_img[:2, :2])

print('\n---------------\n')

cv2.imwrite("label.png", ann_img[:2, :2])
print(cv2.imread("label.png"))
```

运行结果：

``` xml
(4, 3)
[[255 255   1]
 [255 255   1]
 [  1   1   1]
 [  1   1   1]]

---------------

[[255 255]
 [255 255]]

---------------

[[[255 255 255]
  [255 255 255]]

 [[255 255 255]
  [255 255 255]]]
```

来分析下，先看 `print(ann_img[:2, :2])` 这行代码打印出来：

``` xml
[[255 255]
 [255 255]]
```

可以看出是两个维度的，且像素值是赋值的 255，没啥问题。然后 `cv2.imwrite("label.png", ann_img[:2, :2])` 磁盘写入并输出了 `label.png` 图像，然后我们再 `cv2.imread("label.png")` 读取和打印该图像像素值，结果：

``` xml
[[[255 255 255]
  [255 255 255]]

 [[255 255 255]
  [255 255 255]]]
```

可以看到维度由原来的 2 个维度变为 3 维的了，并且**第三个维度的值和前面维度的值是一样的**。

### 给lable上色

代码：

``` python
import numpy as np
import cv2

# 给标签图上色

def color_annotation(label_path, output_path):
   '''
    给class图上色
    '''
    img = cv2.imread(label_path,cv2.CAP_MODE_GRAY)
    color = np.ones([img.shape[0], img.shape[1], 3])
    
    color[img==0] = [255, 255, 255] #其他，白色，0
    color[img==1] = [0, 255, 0]     #植被，绿色，1
    color[img==2] = [0, 0, 0]       #道路，黑色，2
    color[img==3] = [131, 139, 139] #建筑，黄色，3
    color[img==4] = [139, 69, 19]   #水体，蓝色，4

    cv2.imwrite(output_path,color)
```



## 2. scikit-image 的使用





