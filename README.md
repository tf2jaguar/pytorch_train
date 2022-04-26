<div align="center">  
    <p>
        <a href="https://jelly54.github.io"><img src="https://badgen.net/badge/jelly54/read?icon=sourcegraph&color=4ab8a1" alt="read" /></a>
        <img src="https://badgen.net/github/stars/jelly54/pytorch_train?icon=github&color=4ab8a1" alt="stars" />
        <img src="https://badgen.net/github/forks/jelly54/pytorch_train?icon=github&color=4ab8a1" alt="forks" />
        <img src="https://badgen.net/github/open-issues/jelly54/pytorch_train?icon=github" alt="issues" />
    </p>
</div>

# pytorch_train

## 模型训练

Github 地址：[pytorch_train](https://github.com/jelly54/pytorch_train)

![UtyvKf.png](https://s1.ax1x.com/2020/07/14/UtyvKf.png)

训练模型主要分为五个模块：启动器、自定义数据加载器、网络模型、学习率/损失率调整以及训练可视化。

启动器是项目的入口，通过对启动器参数的设置，可以进行很多灵活的启动方式，下图为部分启动器参数设置。

![UtciwD.png](https://s1.ax1x.com/2020/07/14/UtciwD.png)

任何一个深度学习的模型训练都是离不开数据集的，根据多种多样的数据集，我们应该使用一个方式将数据集用一种通用的结构返回，方便网络模型的加载处理。

![Utc9OK.png](https://s1.ax1x.com/2020/07/14/Utc9OK.png)

这里使用了残差网络Resnet-34，代码中还提供了Resnet-18、Resnet-50、Resnet-101以及Resnet-152。残差结构是通过一个快捷连接，极大的减少了参数数量，降低了内存使用。

以下为残差网络的基本结构和Resnet-34 部分网络结构图。

![UtcPeO.png](https://s1.ax1x.com/2020/07/14/UtcPeO.png)

![Utcn6P.png](https://s1.ax1x.com/2020/07/14/Utcn6P.png)


除了最开始看到的train-val图表、Top-、Top-5的error记录表以外，在训练过程中，使用进度条打印当前训练的进度、训练精度等信息。打印时机可以通过上边提到的 启动器 优雅地配置。

![Utc3kQ.png](https://s1.ax1x.com/2020/07/14/Utc3kQ.png)

以下为最终的项目包架构。

```
pytorch_train
  |-- data                -- 存放读取训练、校验、测试数据路径的txt
  |   |-- train.txt       
  |   |-- val.txt
  |   |-- test.txt
  |-- result              -- 存放最终生成训练结果的目录
  |-- util                -- 模型移植工具
  |-- clr.py              -- 学习率
  |-- dataset.py          -- 自定义数据集
  |-- flops_benchmark.py  -- 统计每秒浮点运算次数
  |-- logger.py           -- 日志可视化
  |-- mobile_net.py       -- 网络模型之一 mobile_net2
  |-- resnet.py           -- 网络模型之一 Resnet系列
  |-- run.py              -- 具体执行训练、测试方法
  |-- start.py            -- 启动器
```

![UtgkuV.png](https://s1.ax1x.com/2020/07/14/UtgkuV.png)


## 模型移植

Github 地址：[pytorch_train/transfor](https://github.com/jelly54/pytorch_train/blob/master/util/transfor.py)


```python
import os

import torch
import torchvision

model_pth = os.path.join("results", "2020-04-27_10-27-17", 'checkpoint.pth.tar')
# 将resnet34模型保存为Android可以调用的文件
mobile_pt = os.path.join("results", "2020-04-27_10-27-17", 'resnet34.pt')
num_class = 13
device = 'cpu'  # 'cuda:0'  # cpu

model = torchvision.models.resnet34(num_classes=num_class)
model = torch.nn.DataParallel(model, [0])
model.to(device=device)

checkpoint = torch.load(model_pth, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

model.eval()  # 模型设为评估模式
# 1张3通道224*224的图片
input_tensor = torch.rand(1, 3, 224, 224)  # 设定输入数据格式
traced_script_module = torch.jit.trace(model.module, input_tensor)  # 模型转化
traced_script_module.save(mobile_pt)  # 保存文件
```


## 启动模型训练

<font color=red>启动前需要确保你已经有了本项目使用的数据集 CompCars</font>

### 重新开始新的训练

```shell script
python start.py --data_root "./data" --gpus 0,1,2 -w 2 -b 120 --num_class 13
```

- --data_root 数据集路径位置
- --gups 使用gpu训练的块数
- -w 为gpu加载自定义数据集的工作线程
- -b 用来gpu训练的 batch size是多少
- --num_class 分类类别数量

### 使用上次训练结果继续训练

```shell script
python start.py --data_root "./data" --gpus 0,1,2 -w 2 -b 120 --num_class 13 --resume "results/2020-04-14_12-36-16"
```

- --data_root 数据集路径位置
- --gups 使用gpu训练的块数
- -w 为gpu加载自定义数据集的工作线程
- -b 用来gpu训练的 batch size是多少
- --num_class 分类类别数量
- --resume 上次训练结果文件夹，可继续上次的训练

### 模型移植

将训练好的模型转换为Android可以执行的模型

```shell script
python transfor.py
```

### 项目定制化

- 找寻自己的数据集
- 需要修改启动脚本中 **--num_class**，模型类别

目前项目中具备很多备注记录，稍加review代码就可以理解，如有不清楚，可以私信询问。

### 鼓励一下

<center class="half">
<img alt="image-qxUDIO" src="https://s1.ax1x.com/2022/04/07/qxUDIO.png" width="200" height="240" /><img src="https://s1.ax1x.com/2022/04/07/qxaHAK.jpg" alt="image-qxUBdK" width="200" height="240" />
</center>

有偿提供全套环境搭建+数据集下载+模型迁移+论文范本+技术指导
