# -*- coding: utf-8 -*-
import os


def listdir(path, list_name):
    """
    遍历path下的所有文件，将文件相对地址存入list_name 返回
    :param path: 将要遍历的路径
    :param list_name: list引用
    :return: 文件相对路径list集合
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path.replace("\\", "/"))
    return list_name


def list2file(list_name, file_name, label):
    """
    将list中的值写入文件
    :param list_name: 集合名称
    :param file_name: 文件名称
    :param label: 该集合所属的标签
    :return: None
    """
    _file = open("record/" + file_name + ".txt", "a", encoding='utf-8')
    for name in list_name:
        _file.writelines(",".join([name, label, "\n"]))
    _file.flush()
    _file.close()


def get_model_id(path):
    return path.split('/')[1]


def get_label_dict(origin_file):
    """
    读取attributes文件将model_id 与 type 组成dict
    :return: model_id 与 type 的dict
    """
    labels = {}
    label_file = open(origin_file, "r")
    while 1:
        data_line = label_file.readline().replace("\n", "")
        if not data_line:
            break
        data = data_line.split(" ")
        labels[data[0]] = data[5]
    return labels


def generate_train_val_test_txt(img_path, origin_file, target_file, labels):
    _origin_file = open(origin_file, "r")
    _target_file = open(target_file, "a")
    while 1:
        data_line = _origin_file.readline().replace("\n", "")
        if not data_line:
            break
        data = data_line.split(",")
        print(data)
        path = data[0]
        model_id = get_model_id(path)
        label = labels[str(model_id)]
        _target_file.writelines(",".join([img_path + path, label + "\n"]))

    _origin_file.close()
    _target_file.close()


# 获取测试文件下的各标签数量
# train:{0: 369, 1: 672, 2: 3438, 3: 6386, 4: 2985, 5: 141, 6: 489, 7: 256, 8: 83, 9: 214, 10: 771, 11: 82, 12: 130}
# val  :{0: 0, 1: 127, 2: 590, 3: 1346, 4: 782, 5: 145, 6: 164, 7: 94, 8: 58, 9: 417, 10: 540, 11: 52, 12: 139}
def read_file(file_path):
    dic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    du_file = open(file_path, "r")
    while 1:
        data_line = du_file.readline().replace("\n", "")
        if not data_line:
            break
        data = data_line.split(",")
        va = int(data[1])
        print(va ,dic.get(va))
        dic[va] = dic.get(va) + 1
    print(dic)


# 缩减训练集数量
def write_file(source_file, target_file, target, count):
    dic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    _file = open(source_file, "r")
    res_file = open(target_file, "a")
    while 1:
        data_line = _file.readline().replace("\n", "")
        if not data_line:
            break
        data = data_line.split(",")
        if int(data[1]) == target and dic.get(int(data[1])) < count:
            print(dic.get(int(data[1])))
            res_file.writelines(data_line+"\n")
            dic[int(data[1])] = dic.get(int(data[1])) + 1
        elif dic.get(int(data[1])) == count:
            break



def main():
    label_dict = get_label_dict(r"D:\train\Compcars\data\misc\attributes.txt")
    # 图片路径，campCars分类地址、生成文件地址
    image_location = "./images/"
    generate_train_val_test_txt(image_location, r"D:\train\Compcars\data\train_test_split\classification\train.txt",
                                "../data/train.txt", label_dict)
    generate_train_val_test_txt(image_location, r"D:\train\Compcars\data\train_test_split\classification\test.txt",
                                "../data/test.txt", label_dict)
    generate_train_val_test_txt(image_location, 
                                r"D:\train\Compcars\data\train_test_split\verification\verification_train.txt",
                                "../data/val.txt", label_dict)


if __name__ == '__main__':
    main()
