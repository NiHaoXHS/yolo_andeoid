## 1.YOLO环境准备

安装Anaconda: https://www.anaconda.com/download/success

![conda](./img/conda.png)

![conda1](img\conda1.png)

![conda2](img\conda2.png)

![conda3](img\conda3.png)

![conda4](img\conda4.png)

![conda5](img\conda5.png)

使用conda创建虚拟环境：

![condaenv](img\condaenv.png)

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

![pytorch](img\pytorch.png)

```bash
# pytorch下载
pip3 install torch torchvision
```

安装ultralytics框架:https://docs.ultralytics.com/zh/quickstart

![yolo11](img\yolo11.png)

```bash
pip install -U ultralytics
```

**GPU环境:**

查看cuda最高支持的版本

```bash
nvidia-smi
```

![cuda](img\cuda.png)

CUDA Toolkit下载列表：https://developer.nvidia.com/cuda-toolkit-archive

![cuda1](img\cuda1.png)

下载cuda toolkit （建议本地离线安装）

![cuda2](img\cuda2.png)

安装pytorch https://pytorch.org/get-started/locally/

选择对应的cuda版本

![pytorch1](img\pytorch1.png)

安装ultralytics框架:https://docs.ultralytics.com/zh/quickstart

![yolo11](img\yolo11.png)

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

