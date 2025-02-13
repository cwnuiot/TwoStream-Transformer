import math
import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from numpy import linalg as LA
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
# 定义图片尺寸和背景颜色
width, height = 256, 64  # 256, 64
bg_color = (255, 255, 255)
def get_coor(root,class_num):
    all_stroke,all_label,danbi=[],[],[]
    with open(root, mode="rb") as f:  # f==>*f

      for i in range(class_num):      # 每套都是3755!
        # 1.WordLength 样本内码长度，字节为单位
        WordLength = np.fromfile(f, dtype='uint8', count=1)
        WordCode = np.fromfile(f, dtype='uint8', count=int(WordLength))
        # print(len(WordCode)) # 字内码[176 161]---->  转为 ”啊“中文字
        label=bytes(WordCode).decode('gbk')
        #print(int(label),type(label))
        # 3.PointNum 样本的点的个数
        PointNum = np.fromfile(f, dtype='uint8', count=2)
        PointNum = PointNum[0] + (PointNum[1] << 8)
        # 4.LineNum 样本的笔画个数
        LineNum = np.fromfile(f, dtype='uint8', count=2)
        LineNum = LineNum[0] + (LineNum[1] << 8)
        #print("笔数:{}, 点个数:{}".format(LineNum, PointNum))
        # 5.GetTimePointNum 捕获到时间的点的个数
        GetTimePointNum = np.fromfile(f, dtype='uint8', count=2)
        GetTimePointNum = GetTimePointNum[0] + (GetTimePointNum[1] << 8)
        # print(GetTimePointNum)

        # 6.GetTimePointIndex 捕获到时间的点的序号
        GetTimePointIndex = np.fromfile(f, dtype='uint8', count=GetTimePointNum * 2)
        # print(GetTimePointIndex)
        # 7.ElapsedTime 捕获到时间的点的序号
        ElapsedTime = np.fromfile(f, dtype='uint8', count=GetTimePointNum * 4)
        # ***********************坐标点StrokeData*****************************
        Sample_stroke,Sample_danbi_stroke=[],[]
        for stroke_idx in range(LineNum):       # 笔画数
            StrokePointNum = np.fromfile(f, dtype='uint8', count=2)
            StrokePointNum = sum([j << (i * 8) for i, j in enumerate(StrokePointNum)])
            # StrokePointNum   《=》  一笔划有多少个点
            X = []
            Y = []
            for point_idx in range(StrokePointNum):     # 每一笔画
                Coordinates_x = np.fromfile(f, dtype='uint8', count=2)
                Coordinates_x = sum([j << (i * 8) for i, j in enumerate(Coordinates_x)])
                Coordinates_y = np.fromfile(f, dtype='uint8', count=2)
                Coordinates_y = sum([j << (i * 8) for i, j in enumerate(Coordinates_y)])
                X.append(round(0.1 * Coordinates_x, 2))  # 0~130
                Y.append(round(-0.1 * Coordinates_y, 2))
            #plt.plot(X, Y)
            #print(X,Y)
            for j in range(len(X)):
                Sample_stroke.append([X[j],Y[j]])
            X_Y=[[X[j],Y[j]] for j in range(len(X))]
            Sample_danbi_stroke.append(X_Y)
        # plt.show()
        all_stroke.append(Sample_stroke),all_label.append(label),danbi.append(Sample_danbi_stroke)
    return all_stroke,all_label,danbi



def read_idx(path):
    with open(path, mode="rb") as f:  # f==>*f
        SampleSum = np.fromfile(f, dtype='uint8', count=4)  # 函数读回数据时需要用户指定元素类型，并对数组的形状进行适当的修改
        SampleSum = SampleSum[0] + (SampleSum[1] << 8) + (SampleSum[2] << 16) + (SampleSum[3] << 24)
    return SampleSum

def get_dat(path):
    dat_files = glob.glob(os.path.join(path, "*.dat"))
    idx_files = glob.glob(os.path.join(path, "*.idx"))[0]
    class_num=read_idx(idx_files)
    return dat_files,class_num



def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def removeRedundantPoints(image):
    all_values = [value for point in image for value in point]
    max_value = max(all_values)
    new_stroke_data = [image[0]]
    for i in range(1, len(image)):
        distance = eucliDist(new_stroke_data[-1], image[i])
        if distance >= 0.015*max_value:
            new_stroke_data.append(image[i])
    print(len(image),len(new_stroke_data))
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
def normalize_trajectory(traj):
    traj = removeRedundantPoints(traj)
    traj = normalize(traj)
    return traj
def draw_character_image(trajs):
    trajs = removeRedundantPoints(trajs)
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
def write2txt(dat_files,write2path='Couch_GB2_TXT'):
    dat_files=dat_files[:]
    for i in dat_files:
        name = i.split("/")[-1][:-4]
        file_path=os.path.join(write2path,name)
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
            print('正在创建文件夹')
        all_stroke,all_label,danbi=get_coor(i,class_num)
        if len(all_label)!=len(all_stroke):
            break
        for data in range(len(all_stroke)):
            traj=all_stroke[data]
            txtfile_name='Letter_'+str(data)+'.txt'
            txt_path=os.path.join(file_path,txtfile_name)
            f = open(txt_path, "w")
            f.write(str(traj))
            f.close()
            '''traj=normalize_trajectory(traj)                      #Show Character Image
            image_array = draw_character_image(traj)
            plt.switch_backend('TkAgg')

            # 根据数组绘制图像
            plt.imshow(image_array, cmap='gray')
            plt.axis('on')  # 关闭坐标轴
            plt.show()'''




def write2image(dat_files,write2path='Couch_GB2_IAMGE'):
    jishu=0
    print(dat_files)
    dat_files=dat_files[:]
    for i in dat_files:
        print(jishu)
        jishu=jishu+1
        be_flag=False
        name = i.split("/")[-1][:-4]
        file_path=os.path.join(write2path,name)
        if os.path.exists(file_path):
            imageinfile=os.listdir(file_path)
            if len(imageinfile)==3755:
                be_flag=True
        else:
            os.makedirs(file_path)
            print('正在创建文件夹')
        all_stroke,all_label,danbi=get_coor(i,class_num)
        if len(all_label)!=len(all_stroke):
            print('存在异常')
            break


dat_files,class_num=get_dat('Couch_Letter_195')
write2txt(dat_files)
#write2image(dat_files)



































