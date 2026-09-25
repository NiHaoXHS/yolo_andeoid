## 1.YOLO环境准备

安装Anaconda: https://www.anaconda.com/download/success

![conda](./img/conda.png)

![conda1](./img/conda1.png)

![conda2](./img/conda2.png)

![conda3](./img/conda3.png)

![conda4](./img/conda4.png)

![conda5](./img/conda5.png)

使用conda创建虚拟环境：

![condaenv](./img/condaenv.png)

```bash
# 创建虚拟环境
conda create -n yolo11 python=3.12
# 激活yolo11环境
conda activate yolo11
# 配置pip清华源
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

yolo可以使用cpu或gpu训练模型，cpu训练特别慢，gpu需要使用nvidia显卡

**CPU环境:**

安装pytorch https://pytorch.org/get-started/locally/

![pytorch](./img/pytorch.png)

```bash
# pytorch下载
pip3 install torch torchvision
```

安装ultralytics框架:https://docs.ultralytics.com/zh/quickstart

![yolo11](./img/yolo11.png)

```bash
pip install -U ultralytics
```

**GPU环境:**

查看cuda最高支持的版本

```bash
nvidia-smi
```

![cuda](./img/cuda.png)

CUDA Toolkit下载列表：https://developer.nvidia.com/cuda-toolkit-archive

![cuda1](./img/cuda1.png)

下载cuda toolkit （建议本地离线安装）

![cuda2](./img/cuda2.png)

安装pytorch https://pytorch.org/get-started/locally/

选择对应的cuda版本

![pytorch1](./img/pytorch1.png)

安装ultralytics框架:https://docs.ultralytics.com/zh/quickstart

![yolo11](./img/yolo11.png)

```bash
pip install -U ultralytics
```



## 2.数据集制作

文件夹结构

```cmd
+---data
|   +---images     # 图片文件夹存放图片，例如：1.jpg,1.png
|   |   +---test   # 测试图片
|   |   +---train  # 训练图片
|   |   \---val    # 验证图片
|   \---labels     # 图片标注的标签,例如:label.txt
|       +---test   # 测试图片的标签
|       +---train  # 训练图片的标签
|       \---val    # 验证图片的标签
```

收集图片

例如：收集猫的照片和狗的照片放到一个文件夹img

使用开源工具“**[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)**”对图片手动标注或自动标注

![xanylabling](./img/xanylabling.JPG)

访问地址：https://github.com/CVHub520/X-AnyLabeling

![xanylabingdown](./img/xanylabingdown.png)



![anylabeldown1](./img/anylabeldown1.png)

如果想通过pin，源码安装xanylabeling可以参考：https://xanylabeling.com/zh-Hans/docs/x-anylabeling/get_started

下载好后：

![anylabel](./img/anylabel.png)

双击打开既可以打开软件。（首次启动可能有点慢）

![label](./img/label.png)

![biaozhu1](./img/biaozhu1.png)

![biaozhu2](./img/biaozhu2.png)

查看标签数量：

![gj](./img/gj.png)

![tj](./img/tj.png)

![dc](./img/dc.png)

导出标签和图片：

先在电脑桌面新建一个文本文档class.txt![class](./img/class.png)

双击打开填写:

![img](D:\Users\xiaoming\Desktop\yolo_andeoid\img\img.png)

回到xanylabeling

![dc1](D:\Users\xiaoming\Desktop\yolo_andeoid\img\dc1.png)

![dc3](./img/dc3.png)

![dc4](./img/dc4.png)

![dc5](./img/dc5.png)

导出后：

![dch](./img/dch.png)

点开dog的标签

![bqd](./img/bqd.png)
