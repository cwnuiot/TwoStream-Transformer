import math
import os
import torch
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def calculate_vt(points, current_length):
    vectors = []

    for t in range(len(points)):
        # 初始化所有值为 0
        delta_x = delta_y = sin_alpha_t = cos_alpha_t = sin_beta_t = cos_beta_t = 0
        x_t = y_t = 0
        if t < current_length:
            # 获取当前点的坐标
            x_t, y_t = points[t]

            # 计算 delta_x 和 delta_y
            if t >= 1:
                x_t_minus_1, y_t_minus_1 = points[t-1]
                delta_x = x_t - x_t_minus_1
                delta_y = y_t - y_t_minus_1

            # 计算 alpha_t 的 sin 和 cos
            if t >= 2:
                x_t_minus_2, y_t_minus_2 = points[t-2]
                alpha_t = math.atan2(y_t_minus_1 - y_t_minus_2, x_t_minus_1 - x_t_minus_2)
                sin_alpha_t = math.sin(alpha_t)
                cos_alpha_t = math.cos(alpha_t)

            # 计算 beta_t 的 sin 和 cos
            if t < current_length - 2:
                x_t_plus_1, y_t_plus_1 = points[t+1]
                x_t_plus_2, y_t_plus_2 = points[t+2]
                beta_t = math.atan2(y_t_plus_2 - y_t_plus_1, x_t_plus_2 - x_t_plus_1)
                sin_beta_t = math.sin(beta_t)
                cos_beta_t = math.cos(beta_t)
        # 将计算结果添加到列表中
        vt = [delta_x, delta_y, sin_alpha_t, cos_alpha_t, sin_beta_t, cos_beta_t, x_t, y_t]
        vectors.append(vt)
    return vectors

def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))

def removeRedundantPoints(image):
    all_values = [value for point in image for value in point]
    max_value = max(all_values)
    new_stroke_data = [image[0]]
    for i in range(1, len(image)):
        distance = eucliDist(new_stroke_data[-1], image[i])
        if distance > 0.015*max_value:
            new_stroke_data.append(image[i])
    return new_stroke_data
def normalize(image):
    Lt_all, Ptx_all, Pty_all = 0, 0, 0
    # 计算总长度 Lt 和积分 Ptx, Pty
    for i in range(len(image) - 1):
        pt, pt1, qt, qt1 = image[i][0], image[i + 1][0], image[i][1], image[i + 1][1]
        Lt_now = math.sqrt((pt1 - pt) ** 2 + (qt1 - qt) ** 2)
        Ptx_all += Lt_now * (pt + pt1) / 2
        Pty_all += Lt_now * (qt + qt1) / 2
        Lt_all += Lt_now

    if Lt_all == 0:
        Lt_all = 1

    # 计算归一化中心点 (ux, uy)
    ux = Ptx_all / Lt_all
    uy = Pty_all / Lt_all

    # 计算归一化因子 δx
    X_total = 0
    for i in range(len(image) - 1):
        pt, pt1, qt, qt1 = image[i][0], image[i + 1][0], image[i][1], image[i + 1][1]
        Lt_now = math.sqrt((pt1 - pt) ** 2 + (qt1 - qt) ** 2)
        X_total += Lt_now * ((pt1 - ux) ** 2 + (pt1 - ux) * (pt - ux) + (pt - ux) ** 2) / 3

    if X_total == 0:
        X_total = 1

    delta_x = math.sqrt(X_total / Lt_all)

    # 归一化每个点
    normalized_image = []
    for pt, qt in image:
        x_new = (pt - ux) / delta_x
        y_new = (qt - uy) / delta_x
        normalized_image.append([x_new, y_new])

    return normalized_image
def fix_list_length(points, target_length=140):
    """
    Fix the length of a list of coordinate points to a target length.

    Parameters:
    points (list of lists or tuples): List of (x, y) coordinates.
    target_length (int): The desired length of the list. Default is 140.

    Returns:
    list of lists: A list of (x, y) coordinates fixed to the target length.
    """
    current_length = len(points)
    if current_length < target_length:
        # Calculate the number of [0, 0] points needed to pad the list
        padding_needed = target_length - current_length
        # Extend the list with [0, 0] points
        points.extend([[0, 0]] * padding_needed)
    elif current_length > target_length:
        # Truncate the list to the target length
        points = points[:target_length]
    return points,current_length


def draw_character_image(trajs):
    trajs = np.array(trajs)
    # 使用 Agg 后端
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(2.24, 2.24))
    plt.axis('off')
    plt.plot(trajs[:, 0], trajs[:, 1], marker=None, color='black')
    # 保存图像到 BytesIO 对象
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    # 读取图像数据并转换为数组
    image = Image.open(buf)
    image = image.resize((64, 64), Image.LANCZOS)
    image = image.convert('L')
    # 转换为数组并归一化
    image_array = np.array(image) / 255.0
    # 清理
    buf.close()
    plt.cla()
    plt.close(fig)
    return image_array

def savedata(root, test_size=0.2, random_seed=999):
    characterdata,imagedata,label=[],[],[]
    filenamelist= sorted(os.listdir(root))
    for filename in filenamelist:
        classname = os.listdir(os.path.join(root, filename))
        for name in classname:
            with open(os.path.join(root, filename, name), "r") as f:
                txt = eval(f.read())
                txt = removeRedundantPoints(txt)
                txt=normalize(txt)
                points=np.array(txt)
                txt,current_length = fix_list_length(txt)
                txt=calculate_vt(txt,current_length)
                image_array=draw_character_image(points)
            characterdata.append(txt)
            imagedata.append(image_array)
            number = name.split('_')[-1][:-4]
            label.append(int(number))
            #label.append(int(filename))
    torch.manual_seed(random_seed)
    char_train, char_test, img_train, img_test, label_train, label_test = train_test_split(
        characterdata, imagedata, label, test_size=test_size, random_state=random_seed
    )
    char_train_tensor = torch.tensor(char_train, dtype=torch.float32)
    char_test_tensor = torch.tensor(char_test, dtype=torch.float32)
    img_train_tensor = torch.tensor(img_train, dtype=torch.float32)
    img_test_tensor = torch.tensor(img_test, dtype=torch.float32)
    label_train_tensor = torch.tensor(label_train, dtype=torch.int64)
    label_test_tensor = torch.tensor(label_test, dtype=torch.int64)
    return [char_train_tensor, img_train_tensor, label_train_tensor], [
    char_test_tensor, img_test_tensor, label_test_tensor]

traindir = "Couch_GB2_TXT/Couch_Letter_195"
savedata(traindir)

