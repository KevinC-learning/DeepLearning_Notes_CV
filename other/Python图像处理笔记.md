# 0、前言和目录

基于 python 脚本语言开发的数字图片处理包，比如 PIL、Pillow、opencv、scikit-image 等。

- PIL 和 Pillow 只提供最基础的数字图像处理，功能有限；

- opencv 实际上是一个 c++ 库，只是提供了 python 接口，更新速度非常慢。
- scikit-image 是基于 scipy 的一款图像处理包，它将图片作为 numpy 数组进行处理，正好与 matlab 一样，因此，我们最终选择 scikit-image 进行数字图像处理。

学习：[python skimage图像处理(一) - 简书](<https://www.jianshu.com/p/f2e88197e81d>)

## 目录

[一、opencv-python 的使用](#一opencv-python-的使用)

[二、scikit-image 的使用](#二scikit-image的使用)





---

# 一、opencv-python 的使用

## 1. opencv-python安装

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



## 2. opencv-python 图像处理

### OpenCV API 详解

#### 1）cv2.imread() 和cv2.imwrite()

（1）

imread 的函数原型是：Mat imread( const string& filename, int flags=1 );

Mat是OpenCV里的一个数据结构，在这里我们定义一个Mat类型的变量img，用于保存读入的图像，在本文开始有写到，我们用imread函数来读取图像，第一个字段标识图像的文件名（包括扩展名），第二个字段用于指定读入图像的颜色和深度，它的取值可以有以下几种：

1) CV_LOAD_IMAGE_UNCHANGED (<0)，以原始图像读取（包括alpha通道），

2) CV_LOAD_IMAGE_GRAYSCALE ( 0)，以灰度图像读取

3) CV_LOAD_IMAGE_COLOR (>0)，以RGB格式读取

——from：<https://blog.csdn.net/zhangpinghao/article/details/8144829>

文档中是这么写的：

``` markdown
Flags specifying the color type of a loaded image:

CV_LOAD_IMAGE_ANYDEPTH - If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
CV_LOAD_IMAGE_COLOR - If set, always convert image to the color one
CV_LOAD_IMAGE_GRAYSCALE - If set, always convert image to the grayscale one
>0 Return a 3-channel color image.
Note

In the current implementation the alpha channel, if any, is stripped from the output image. Use negative value if you need the alpha channel.

=0 Return a grayscale image.
<0 Return the loaded image as is (with alpha channel).
```

大致翻译一下：

Flags指定了所读取图片的颜色类型

- CV_LOAD_IMAGE_ANYDEPTH 返回图像的深度不变。

- CV_LOAD_IMAGE_COLOR 总是返回一个彩色图。

- CV_LOAD_IMAGE_GRAYSCALE 总是返回一个灰度图。

0 返回3通道彩色图，注意：alpha 通道将被忽略，如果需要alpha 通道，请使用负值

=0 返回灰度图

<0 返回原图（带 alpha 通道）

我觉得这里最大的问题就是一会说深度，一会说通道数，两个问题都没有说明白。

实测，当读取一副黑白图时，如果使用Flags=2（CV_LOAD_IMAGE_ANYDEPTH），此时Flags>0，得到的仍是黑白图而不是彩色图。其它的值，如 1,3,4 等均是彩色。

所以我觉得第一句话应该改为  CV_LOAD_IMAGE_ANYDEPTH 返回图像原有的深度，但是通道数变为 1，这是 Flags>0 中的特例

自己测了一下，然后总结如下：

- flag=-1 时，8位深度，原通道
- flag=0，8位深度，1通道
- flag=1,   8位深度  ，3通道
- flag=2，原深度，1通道
- flag=3,  原深度，3通道
- flag=4，8位深度 ，3通道

在源码中可以看到。默认是 1。

——from：[opencv中imread第二个参数的含义](<https://blog.csdn.net/z914022466/article/details/52709981>)

**!!!注意：** cv2.imread() 和cv2.imwrite() 函数，其中第一个参数 finename，一定是在已存在的目录，若指定的是不存在的目录，就不会写入和读取到图像文件了。

（2）

注1：本人使用 labelme 进行进行标注得到 json 文件，然后使用 `labelme_json_to_dataset` 转换的时候，得到的 label.png 为彩色，而非黑色图像，看评论有人说是版本问题… 

注2：然后我安装了 labelme 旧版 `2.9.0`，`pip install labelme==2.9.0`，发现这个版本 `labelme_json_to_dataset` 命令生成的 `label.png` 文件确实是全黑色，并且是 **16 位深度**的。

然后我使用 cv2.imread(“label.png”) 读取发现得到的数值最小最大都是0；使用 cv2.imread(label.png”, 2) 读取发现得到的数值最小是0，最大是1，为什么呢？后来知道了。先看 [opencv imread()方法第二个参数介绍](<https://blog.csdn.net/qq_27278957/article/details/84589526>) | [opencv中imread第二个参数的含义](<https://blog.csdn.net/z914022466/article/details/52709981#>)，可以说，`imread(const string& filename, int flag=1)`，filename 指图像名称，flag 指读取图像颜色类型。

- flag=-1时，8位深度，原通道
- flag=0，8位深度，1通道
- flag=1,   8位深度  ，3通道
- flag=2，原深度，1通道
- flag=3,  原深度，3通道
- flag=4，8位深度 ，3通道

我解释下：因为 label.png 是 16 位的，默认 flag=1，按上可以看到只读取到了图像的 8 位，得到 3 通道，得到的全是 0；若 flag=2，按原深度即读取了图像位深度 16 位，得到了数值 1。

我的理解：本质原因在于 imread 读取了图像的多少位。另外注意，如果本来是 1 个通道的图像，imread 第二个参数选择了返回 3 个通道的，那么第一个通道读取的数值，在相同像素的位置另外两个通道也会有同样数值。

按我自己理解我总结下，不做参考用，仅供自己看：

- 如果打开电脑上图像的属性，看到深度 8 位，你用「8位深度，1通道」读取，得到单通道图，你用「8位深度，3通道」，得到3通道图像数值（在另外两通道数值是相同的）；
- 如果图像属性是 16 位的，你用「8位深度，1通道」读取，只能读取8位长度的数据，单通道，你用「原深度，1通道」你能读取16位长度，单通道，你用「原深度，3通道」你能读取16位长度数据，得到三通道（在另外两个通道数值是相同的）
- 如果图像属性是 32 位的，你用「8位深度，3通道」，得到是原图，你用「8位深度，1通道」读取，转灰度图，有个彩色图RGB数值转灰度图数值的计算公式：Grey = 0.299\*R + 0.587\*G + 0.114\*B，按计算公式得到灰度图数值，你用「原深度，1通道」读取，得到和以「8位深度，1通道」读取一样的图像。

> 另外，建议可以使用  matlab 软件 imread(imagepath) 读取图像，点击打开工作区的 ans，可以看到图像数值以及是通道数量。

（暂时理解到这个进步。。。以后在探究。。。



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

**灰度图像**是用不同饱和度的黑色来表示每个图像点，比如用8位 0-255数字表示“灰色”程度，每个像素点只需要一个灰度值，8 位即可，这样一个 3X3 的灰度图，只需要9个byte就能保存

RGB 值和灰度的转换，实际上是人眼对于彩色的感觉到亮度感觉的转换，这是一个心理学问题，有一个公式：

**Grey = 0.299\*R + 0.587\*G + 0.114\*B**

根据这个公式，依次读取每个像素点的 R，G，B 值，进行计算灰度值（转换为整型数），将灰度值赋值给新图像的相应位置，所有像素点遍历一遍后完成转换。

——from：[RGB图像转为灰度图](https://blog.csdn.net/u010312937/article/details/71305714)

### 代码生成的图像

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

> 这里提一下，我发现经过 cv.imread() 读取的图像，打印出来的三个维度的数值，是按照  BGR 顺序打印的。在使用 cv.imwrite() 写入输出图像的时候，第二个参数也得按照 BGR 顺序存储，所以如果label上色，记得按照 BGR 顺序赋值。
>
> 参考：[opencv使用BGR而非RGB的原因](<https://blog.csdn.net/weixin_35653315/article/details/73460022>)  
>
> ```
> label_img = cv2.imread("./aaa.png")
> label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
> ```
>
> 注：可以使用颜色空间转换函数 cv2.cvtColor 设置 cv2 的默认读取和写入通道顺序。关于该函数讲解见：[opencv中颜色空间转换函数 cv2.cvtColor()](<https://blog.csdn.net/u012193416/article/details/79312798>)

### label图像上色

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
img = cv2.imread("label.png")
print(img.shape)
print(img)
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

(2, 2, 3)
[[[255 255 255]
  [255 255 255]]

 [[255 255 255]
  [255 255 255]]]
```

可以看到本地保存下来的 label.png 信息，是 8bit 的：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190617173609.png)

我们来分析下，先看 `print(ann_img[:2, :2])` 这行代码打印出来：

``` xml
[[255 255]
 [255 255]]
```

可以看出是两个维度的，且像素值是赋值的 255，没啥问题。然后 `cv2.imwrite("label.png", ann_img[:2, :2])` 磁盘写入并保存 `label.png` 到本地，然后我们再 `cv2.imread("label.png")` 读取和打印该图像 shape 和像素值，结果：

``` xml
(2, 2, 3)
[[[255 255 255]
  [255 255 255]]

 [[255 255 255]
  [255 255 255]]]
```

可以看到维度由原来的 2 个维度变为 3 维的了，并且**第三个维度的值和前面维度的值是一样的**。

> 解释：本质是因 img = cv2.imread("label.png")  默认以第二个参数为 flags=1 方式读取的（关于第二个参数的详解参考前面的内容）。改为 img = cv2.imread("label.png", 0) 读取，可以看到结果如下：
>
> ``` python
> (2, 2)
> [[[255 255]
>   [255 255]]
> ```

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

**注意：**这里赋值顺序是先 BGR 顺序，即 [139, 69, 19] 赋值的分别是 B 通道、G 通道、R 通道。具体原因网上查找下资料。

### 图像归一化：/255.0 和/127.5 -1

在代码中看到图像的2种处理方式：

- img/255.0
- img/127.5 - 1

第一种是对图像进行归一化，范围为[0, 1]，第二种也是对图像进行归一化，范围为[-1, 1]，这两种只是归一化范围不同，为了直观的看出2种区别，分别对图像进行两种处理：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190505170110.png)

从图中可以看出， 第二种方式图像显示的更黑，其直方图如下：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190505170130.png)

同样，其直方图的分布规律相同，第二种分布相对稀疏。——from：[图像处理/255.0 和/127.5 -1](<https://blog.csdn.net/u011276025/article/details/76050377>)



## 3. 图像数据类型及转换

在 skimage 中，一张图片就是一个简单的 numpy 数组，数组的数据类型有很多种，相互之间也可以转换。这些数据类型及取值范围如下表所示：

| Data type | Range             |
| --------- | ----------------- |
| uint8     | 0 to 255          |
| uint16    | 0 to 65535        |
| uint32    | 0 to 232          |
| float     | -1 to 1 or 0 to 1 |
| int8      | -128 to 127       |
| int16     | -32768 to 32767   |
| int32     | -231 to 231 - 1   |

一张图片的像素值范围是 [0, 255]，因此默认类型是 unit8。可用如下代码查看数据类型：

``` python
from skimage import io,data
img=data.chelsea()
print(img.dtype.name)
```

在上面的表中，特别注意的是 float 类型，它的范围是 [-1,1] 或 [0,1] 之间。一张彩色图片转换为灰度图后，它的类型就由 unit8 变成了 float。

——from：[python数字图像处理（4）：图像数据类型及颜色空间转换](https://www.cnblogs.com/denny402/p/5122328.html)





# 二、scikit-image 的使用







---

# 三、libtiff.TIFF

##  python下tiff图像的读取和保存方法

对比测试 **scipy.misc** 和 **PIL.Image** 和 **libtiff.TIFF** 三个库

测试两类输入矩阵：

1. (读取图像) 读入uint8、uint16、float32的lena.tif
2. (生成矩阵) 使用numpy产生随机矩阵，float64的mat

``` python
import numpy as np
from scipy import misc
from PIL import Image
from libtiff import TIFF 
#
# 读入已有图像,数据类型和原图像一致
tif32 = misc.imread('.\test\lena32.tif') #<class 'numpy.float32'>
tif16 = misc.imread('.\test\lena16.tif') #<class 'numpy.uint16'>
tif8  = misc.imread('.\test\lena8.tif')  #<class 'numpy.uint8'>
# 产生随机矩阵,数据类型float64
np.random.seed(12345)
flt = np.random.randn(512, 512)          #<class 'numpy.float64'>
# 转换float64矩阵type,为后面作测试
z8 = (flt.astype(np.uint8))              #<class 'numpy.uint8'>
z16 = (flt.astype(np.uint16))            #<class 'numpy.uint16'>
z32 = (flt.astype(np.float32))           #<class 'numpy.float32'> 
```

①对图像和随机矩阵的存储

``` python
# scipy.misc『不论输入数据是何类型，输出图像均为uint8』
misc.imsave('.\test\lena32_scipy.tif', tif32)   #--> 8bit(tif16和tif8同)

misc.imsave('.\test\\randmat64_scipy.tif', flt) #--> 8bit
misc.imsave('.\test\\randmat8_scipy.tif', z8)   #--> 8bit(z16和z32同)

# PIL.Image『8位16位输出图像与输入数据类型保持一致，64位会存成32位』
Image.fromarray(tif32).save('.\test\lena32_Image.tif') #--> 32bit
Image.fromarray(tif16).save('.\test\lena16_Image.tif') #--> 16bit
Image.fromarray(tif8).save('.\test\lena8_Image.tif')   #--> 8bit

Image.fromarray(flt).save('.\test\\randmat_Image.tif') #--> 32bit(flt.min~flt.max)
im = Image.fromarray(flt.astype(np.float32))                      
im.save('.\test\\randmat32_Image.tif')                 #--> 32bit(灰度值范围同上)
#『uint8和uint16类型转换,会使输出图像灰度变换到255和65535』
im = Image.frombytes('I;16', (512, 512), flt.tostring())
im.save('.\test\\randmat16_Image1.tif')                #--> 16bit(0~65535)
im = Image.fromarray(flt.astype(np.uint16))                      
im.save('.\test\\randmat16_Image2.tif')                #--> 16bit(0~65535)
im = Image.fromarray(flt.astype(np.uint8))                      
im.save('.\test\\randmat8_Image.tif')                  #--> 8bit(0~255)

# libtiff.TIFF『输出图像与输入数据类型保持一致』
tif = TIFF.open('.\test\\randmat_TIFF.tif', mode='w') 
tif.write_image(flt, compression=None)
tif.close() #float64可以存储,但因BitsPerSample=64,一些图像软件不识别
tif = TIFF.open('.\test\\randmat32_TIFF.tif', mode='w') 
tif.write_image(flt.astype(np.float32), compression=None)
tif.close() #--> 32bit(flt.min~flt.max)
#『uint8和uint16类型转换,会使输出图像灰度变换到255和65535』
tif = TIFF.open('.\test\\randmat16_TIFF.tif', mode='w') 
tif.write_image(flt.astype(np.uint16), compression=None)
tif.close() #--> 16bit(0~65535,8位则0~255)
```

②图像或矩阵归一化对存储的影响

``` python
# 『使用scipy,只能存成uint8』
z16Norm = (z16-np.min(z16))/(np.max(z16)-np.min(z16))  #<class 'numpy.float64'>
z32Norm = (z32-np.min(z32))/(np.max(z32)-np.min(z32))
scipy.misc.imsave('.\test\\randmat16_norm_scipy.tif', z16Norm)  #--> 8bit(0~255)

# 『使用Image,归一化后变成np.float64 直接转8bit或16bit都会超出阈值,要*255或*65535』
# 『如果没有astype的位数设置,float64会直接存成32bit』
im = Image.fromarray(z16Norm)
im.save('.\test\\randmat16_norm_Image.tif')       #--> 32bit(0~1)
im = Image.fromarray(z16Norm.astype(np.float32))
im.save('.\test\\randmat16_norm_to32_Image.tif')  #--> 32bit(灰度范围值同上)
im = Image.fromarray(z16Norm.astype(np.uint16))
im.save('.\test\\randmat16_norm_to16_Image.tif')  #--> 16bit(0~1)超出阈值
im = Image.fromarray(z16Norm.astype(np.uint8))
im.save('.\test\\randmat16_norm_to8_Image.tif')   #--> 8bit(0~1)超出阈值

im = Image.fromarray((z16Norm*65535).astype(np.uint16))
im.save('.\test\\randmat16_norm_to16_Image1.tif') #--> 16bit(0~65535)
im = Image.fromarray((z16Norm*255).astype(np.uint16))
im.save('.\test\\randmat16_norm_to16_Image2.tif') #--> 16bit(0~255)
im = Image.fromarray((z16Norm*255).astype(np.uint8))
im.save('.\test\\randmat16_norm_to8_Image2.tif')  #--> 8bit(0~255)
# 『使用TIFF结果同Image』
```

③TIFF读取和存储多帧 tiff 图像

``` python
#tiff文件解析成图像序列：读取tiff图像
def tiff_to_read(tiff_image_name):  
    tif = TIFF.open(tiff_image_name, mode = "r")  
    im_stack = list()
    for im in list(tif.iter_images()):  
        im_stack.append(im)
    return  
    #根据文档,应该是这样实现,但测试中不管是tif.read_image还是tif.iter_images读入的矩阵数值都有问题
  
#图像序列保存成tiff文件：保存tiff图像   
def write_to_tiff(tiff_image_name, im_array, image_num):
    tif = TIFF.open(tiff_image_name, mode = 'w') 
    for i in range(0, image_num):  
        im = Image.fromarray(im_array[i])
        #缩放成统一尺寸  
        im = im.resize((480, 480), Image.ANTIALIAS)  
        tif.write_image(im, compression = None)     
    out_tiff.close()  
    return   
```

补充：libtiff 读取多帧 tiff 图像

因为（单帧）TIFF.open().read_image()和（多帧）TIFF.open().iter_images() 有问题，故换一种方式读

``` python
from libtiff import TIFFfile
tif = TIFFfile('.\test\lena32-3.tif')
samples, _ = tif.get_samples()
```

——from：[python下tiff图像的读取和保存方法](<https://blog.csdn.net/index20001/article/details/80242450>)

## tiff文件的保存与解析

tiff 文件是一种常用的图像文件格式，支持将多幅图像保存到一个文件中，极大得方便了图像的保存和处理。

python 中支持 tiff 文件处理的是 libtiff 模块中的 TIFF 类（libtiff 下载链接<https://pypi.python.org/pypi/libtiff/>）。

这里主要介绍 tiff 文件的解析和保存，具体见如下代码：

``` python
from libtiff import TIFF
from scipy import misc
 
##tiff文件解析成图像序列
##tiff_image_name: tiff文件名；
##out_folder：保存图像序列的文件夹
##out_type：保存图像的类型，如.jpg、.png、.bmp等
def tiff_to_image_array(tiff_image_name, out_folder, out_type): 
          
    tif = TIFF.open(tiff_image_name, mode = "r")
    idx = 0
    for im in list(tif.iter_images()):
		#
        im_name = out_folder + str(idx) + out_type
        misc.imsave(im_name, im)
        print im_name, 'successfully saved!!!'
        idx = idx + 1
    return
 
##图像序列保存成tiff文件
##image_dir：图像序列所在文件夹
##file_name：要保存的tiff文件名
##image_type:图像序列的类型
##image_num:要保存的图像数目
def image_array_to_tiff(image_dir, file_name, image_type, image_num):
 
    out_tiff = TIFF.open(file_name, mode = 'w')
	
	#这里假定图像名按序号排列
    for i in range(0, image_num):
        image_name = image_dir + str(i) + image_type
        image_array = Image.open(image_name)
		#缩放成统一尺寸
        img = image_array.resize((480, 480), Image.ANTIALIAS)
        out_tiff.write_image(img, compression = None, write_rgb = True)
		
    out_tiff.close()
    return 
```

——from：[【python图像处理】tiff文件的保存与解析](<https://blog.csdn.net/guduruyu/article/details/71191709>)

很多医学文件采用格式TIFF格式存储，并且一个TIFF文件由多帧序列组合而成，使用libtiff可以将TIFF文件中的多帧提取出来。

``` xml
from libtiff import TIFF

def tiff2Stack(filePath):
    tif = TIFF.open(filePath,mode='r')
    stack = []
    for img in list(tif.iter_images()):
        stack.append(img)
    return  stack
```

——from：[Python进行TIFF文件处理](<https://www.jianshu.com/p/4db164533d75>)





