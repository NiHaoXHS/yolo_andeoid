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
|   |---labels     # 图片标注的标签,例如:label.txt
|   |   +---test   # 测试图片的标签
|   |   +---train  # 训练图片的标签
|   |   \---val    # 验证图片的标签
	+---dataset.yaml #配置文件
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



## 3.模型训练和测试

在同级别目录下创建脚本:split_dataset.py对数据进行划分并且创建配置文件YAML

![split](./img/split.png)

```python
import os
import shutil
import random
import yaml

# 配置参数
SOURCE_DIR = r"./labels"  # 源数据目录，包含图像和标签文件
OUTPUT_DIR = r"./data"    # 输出目录，将创建train/val/test子目录
TRAIN_RATIO = 0.7  # 训练集比例 70%
VAL_RATIO = 0.2    # 验证集比例 20%
TEST_RATIO = 0.1   # 测试集比例 10%

# 类别名称定义
CLASS_NAMES = {
    0: 'dog',  # 类别0: 狗
    1: 'cat'   # 类别1: 猫
}

def create_directories():
    """创建必要的目录结构"""
    # 定义需要创建的所有目录
    dirs = [
        os.path.join(OUTPUT_DIR, 'images', 'train'),  # 训练集图像目录
        os.path.join(OUTPUT_DIR, 'images', 'val'),    # 验证集图像目录
        os.path.join(OUTPUT_DIR, 'images', 'test'),   # 测试集图像目录
        os.path.join(OUTPUT_DIR, 'labels', 'train'),  # 训练集标签目录
        os.path.join(OUTPUT_DIR, 'labels', 'val'),    # 验证集标签目录
        os.path.join(OUTPUT_DIR, 'labels', 'test'),   # 测试集标签目录
    ]

    # 遍历创建所有目录，exist_ok=True表示目录已存在时不报错
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"已创建目录: {dir_path}")

def get_image_label_pairs():
    """从源目录获取所有图像-标签配对"""
    pairs = []

    # 遍历源目录中的所有文件
    for file in os.listdir(SOURCE_DIR):
        if file.endswith('.txt'):  # 找到标签文件
            base_name = file[:-4]  # 去除.txt后缀，获取基础文件名
            # 查找对应的图像文件，支持多种图像格式
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img_path = os.path.join(SOURCE_DIR, base_name + ext)
                if os.path.exists(img_path):  # 如果找到对应的图像文件
                    label_path = os.path.join(SOURCE_DIR, file)
                    # 将图像和标签配对信息保存到列表中
                    pairs.append({
                        'image': img_path,      # 图像文件路径
                        'label': label_path,    # 标签文件路径
                        'basename': base_name,  # 基础文件名
                        'img_ext': ext          # 图像文件扩展名
                    })
                    break  # 找到匹配的图像后跳出循环

    return pairs

def split_dataset(pairs):
    """将数据集分割为训练集、验证集和测试集"""
    random.seed(42)  # 设置随机种子，保证结果可复现
    random.shuffle(pairs)  # 随机打乱数据

    # 计算各个数据集的分割点
    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)  # 训练集结束位置
    val_end = train_end + int(total * VAL_RATIO)  # 验证集结束位置

    # 按比例分割数据
    train_pairs = pairs[:train_end]  # 训练集
    val_pairs = pairs[train_end:val_end]  # 验证集
    test_pairs = pairs[val_end:]  # 测试集

    return train_pairs, val_pairs, test_pairs

def copy_files(pairs, split_name):
    """将文件复制到相应的目录"""
    for pair in pairs:
        # 复制图像文件到对应的数据集目录
        img_dest = os.path.join(OUTPUT_DIR, 'images', split_name,
                               pair['basename'] + pair['img_ext'])
        shutil.copy2(pair['image'], img_dest)  # copy2保留文件元数据

        # 复制标签文件到对应的数据集目录
        label_dest = os.path.join(OUTPUT_DIR, 'labels', split_name,
                                 pair['basename'] + '.txt')
        shutil.copy2(pair['label'], label_dest)

    print(f"已复制 {len(pairs)} 个文件到 {split_name} 集")

def create_yaml_config():
    """创建YOLO格式的YAML配置文件"""
    # 配置数据集的基本信息
    config = {
        'path': './data',  # 数据集根目录
        'train': 'images/train',  # 训练集图像路径
        'val': 'images/val',      # 验证集图像路径
        'test': 'images/test',    # 测试集图像路径
        'nc': len(CLASS_NAMES),   # 类别数量
        'names': CLASS_NAMES      # 类别名称字典
    }

    yaml_path = os.path.join(OUTPUT_DIR, 'dataset.yaml')

    # 写入YAML文件，使用UTF-8编码支持中文
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nYAML配置文件已保存到: {yaml_path}")
    return yaml_path

def main():
    print("=" * 60)
    print("YOLO 数据集分割工具")
    print("=" * 60)

    print("\n创建目录结构")
    create_directories()

    print("\n查找图像-标签对")
    pairs = get_image_label_pairs()
    print(f"找到 {len(pairs)} 个图像-标签对")

    if len(pairs) == 0:
        print("错误: 未找到图像-标签对!")
        return

    print("\n分割数据集")
    train_pairs, val_pairs, test_pairs = split_dataset(pairs)
    print(f"训练集: {len(train_pairs)} 个样本 ({TRAIN_RATIO*100:.0f}%)")
    print(f"验证集: {len(val_pairs)} 个样本 ({VAL_RATIO*100:.0f}%)")
    print(f"测试集: {len(test_pairs)} 个样本 ({TEST_RATIO*100:.0f}%)")

    print("\n复制文件")
    copy_files(train_pairs, 'train')
    copy_files(val_pairs, 'val')
    copy_files(test_pairs, 'test')

    print("\n创建YAML配置")
    yaml_path = create_yaml_config()

    print("\n" + "=" * 60)
    print("数据集分割完成")
    print("=" * 60)
    print(f"\n数据集位置: {OUTPUT_DIR}")
    print(f"配置文件: {yaml_path}")

if __name__ == "__main__":
    main()
```

运行split_dataset.py后：

可以看到yaml

![yaml](./img/yaml.png)

创建train.py

```python
# 导入YOLO模型库，用于目标检测训练
from ultralytics import YOLO
# 导入PyTorch库，用于检测GPU可用性
import torch

def train_yolo():
    """
    训练YOLO模型的主函数
    用于训练猫狗分类的目标检测模型
    """
    # 加载预训练的YOLO11模型（small版本）
    model = YOLO('yolo11s.pt')

    # 检测并显示使用的训练设备（GPU或CPU）
    if torch.cuda.is_available():
        print(f'使用GPU训练: {torch.cuda.get_device_name(0)}')
    else:
        print('使用CPU训练')

    # 数据集配置文件路径
    data_config = r'./data/dataset.yaml'

    # 开始训练模型
    results = model.train(
        data=data_config,     # 数据集配置文件
        epochs=200,           # 训练轮数：200个epoch
        imgsz=640,            # 输入图像尺寸：640x640像素
        batch=18,             # 批次大小：每批处理18张图片
        workers=2,            # 数据加载的工作进程数
        cache='ram',          # 将数据缓存到内存中以加快训练速度
        # resume=True,        # 从上次中断处恢复训练（需要时取消注释）
    )
    return results

if __name__ == '__main__':
    train_yolo()

```

运行train.py,首次运行训练会自动下载”yolo11s.pt“模型权重

结果：

![result](D:\Users\xiaoming\Desktop\yolo_andeoid\img\result.png)

![train](D:\Users\xiaoming\Desktop\yolo_andeoid\img\train.png)

在同基本目录下创建test.py测试脚本：

```python
from ultralytics import YOLO
import torch

def test_yolo():
    """
    测试YOLO模型的主函数
    用于评估训练好的猫狗检测模型在测试集上的性能
    """
    # 加载训练好的最佳模型
    model = YOLO(r'runs/detect/train/weights/best.pt')

    # 检测并显示使用的设备（GPU或CPU）
    if torch.cuda.is_available():
        print(f'使用GPU测试: {torch.cuda.get_device_name(0)}')
    else:
        print('使用CPU测试')

    # 数据集配置文件路径
    data_config = r'./data/dataset.yaml'

    print('在测试集上评估模型')

    # 在测试集上验证模型
    results = model.val(
        data=data_config,     # 数据集配置文件
        split='test',         # 使用test数据集进行评估
        imgsz=640,            # 输入图像尺寸：640x640像素
        batch=16,             # 批次大小
        save_json=True,       # 保存结果为JSON格式
        save_hybrid=True,     # 保存标签和预测的混合版本
        conf=0.5,            # 置信度阈值
        iou=0.6,              # NMS的IOU阈值
        plots=True,           # 生成评估图表
    )

    # 打印评估结果
    print('\n' + '='*50)
    print('测试结果摘要:')
    print('='*50)
    print(f'mAP50: {results.box.map50:.4f}')
    print(f'mAP50-95: {results.box.map:.4f}')
    print(f'Precision: {results.box.mp:.4f}')
    print(f'Recall: {results.box.mr:.4f}')
    print('='*50)

    return results

if __name__ == '__main__':
    test_yolo()

```

测试结果：

![test1](./img/test1.png)

![test1 (2)](./img/test1 (2).png)
