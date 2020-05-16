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
