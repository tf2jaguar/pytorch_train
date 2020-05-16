# pytorch_train

## 重新开始新的训练

```shell script
python start.py --data_root "./data" --gpus 0,1,2 -w 2 -b 120 --num_class 13
```

## 使用上次训练结果继续训练

```shell script
python start.py --data_root "./data" --gpus 0,1,2 -w 2 -b 120 --num_class 13 --resume "results/2020-04-14_12-36-16"
```

## 将训练好的模型转换为Android可以执行的模型

```shell script
python transfor.py
```