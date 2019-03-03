## 使用指定的 GPU

1. 查看 GPU：nvidia-smi -L
2. 查看 7 号 GPU：nvidia-smi -q -i 7

  - 如果只看 memory 情况，可以用：nvidia-smi -q -i 7 -d MEMORY
3. 设置 GPUid：CUDA_VISIBLE_DEVICES=6（或CUDA_VISIBLE_DEVICES=6,7）command

  - 这条命令适用于命令行运行 tensorflow 程序的时候，指定 GPU
  - 只需要在命令之前设置环境变量，简单来说比如原本程序是命令行运行 python train.py
  - 假定这里 GPU 总共有八块，通过 nvidia-smi 查看发现 5,6,7 是空闲的（从0开始编号）
  - 则运行命令修改为：CUDA_VISIBLE_DEVICES=5,6,7 python train.py
  - 这样在跑你的网络之前，告诉程序只能看到 5、6、7 号 GPU，其他的 GPU 它不可见



---

来源：[tensorflow中使用指定的GPU及GPU显存](https://www.cnblogs.com/darkknightzh/p/6591923.html)

（1）终端执行程序时设置使用的GPU

如果电脑有多个 GPU，tensorflow 默认全部使用。如果想只使用部分 GPU，可以设置CUDA_VISIBLE_DEVICES。在调用 python 程序时，可以使用：

``` xml
CUDA_VISIBLE_DEVICES=1 python my_script.py
```

``` xml
Environment Variable Syntax      Results

CUDA_VISIBLE_DEVICES=1           Only device 1 will be seen
CUDA_VISIBLE_DEVICES=0,1         Devices 0 and 1 will be visible
CUDA_VISIBLE_DEVICES="0,1"       Same as above, quotation marks are optional
CUDA_VISIBLE_DEVICES=0,2,3       Devices 0, 2, 3 will be visible; device 1 is masked
CUDA_VISIBLE_DEVICES=""          No GPU will be visible
```

（2）python 代码中设置使用的 GPU

如果要在 python 代码中设置使用的 GPU（如使用 pycharm 进行调试时），可以使用下面的代码：

``` xml
CUDA_VISIBLE_DEVICES=1 python my_script.py
```



----

来源：[Tensorflow 学习笔记（七） ———— 多GPU操作](https://applenob.github.io/tf_7.html)

Tensorflow 中指定使用设备：

``` xml
"/cpu:0": 机器中的 CPU
"/gpu:0": 机器中的 GPU, 如果你有一个的话.
"/gpu:1": 机器中的第二个 GPU, 以此类推...
```

但是有问题：

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190303154853.png)



## GPU 显存分析

默认 tensorflow 是使用 GPU 尽可能多的显存。可以通过下面的方式，来设置使用的 GPU 显存：

``` xml
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))    
```

上面分配给 tensorflow 的 GPU 显存大小为：GPU 实际显存*0.7。

上面的只能设置固定的大小。如果想按需分配，可以使用`allow_growth`参数（参考网址：<http://blog.csdn.net/cq361106306/article/details/52950081>）：

``` xml
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
```



---

来源：[科普帖：深度学习中GPU和显存分析 - 知乎](https://zhuanlan.zhihu.com/p/31558973)

nvidia-smi 是 Nvidia 显卡命令行管理套件，基于 NVML 库，旨在管理和监控 Nvidia GPU 设备。

命令行输入nvidia-smi 指令即可看到当前 nvidia 显卡的使用情况。

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190303155029.png)

但是有时我们希望不仅知道那一固定时刻的 GPU 使用情况，我们希望一直掌握其动向，此时我们就希望周期性地输出，比如每 10s 就更新显示。 这时候就需要用到 watch命令，来周期性地执行 nvidia-smi 命令了。

- watch的基本用法是：watch [options] command
- 作用：周期性执行某一命令，并将输出显示。
- 最常用的参数是 -n， 后面指定是每多少秒来执行一次命令。
- 监视显存：我们设置为每 10s 显示一次显存的情况：watch -n 10 nvidia-smi
- **测试发现：该 watch 为 Linux 下命令， windows 下不可用**
- **或者直接该命令的一个选项 -l，通过该选项我们可以动态查看 GPU 使用情况: `nvidia-smi.exe -l 10`，表示10秒钟更新一次信息**

显存占用和 GPU 利用率是两个不一样的东西，显卡是由 GPU 计算单元和显存等组成的，显存和 GPU 的关系有点类似于内存和 CPU 的关系。

这里推荐一个好用的小工具：gpustat，直接 `pip install gpustat` 即可安装，gpustat 基于 nvidia-smi，可以提供更美观简洁的展示，结合 watch 命令，可以动态实时监控 GPU 的使用情况。`watch --color -n1 gpustat -cpu `

![](https://img-1256179949.cos.ap-shanghai.myqcloud.com/20190303155232.png)