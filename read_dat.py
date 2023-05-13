import math
import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from numpy import linalg as LA
import glob
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
def removeRedundantPoints(image,thres=0.015):  # 去除冗余点
    if len(image)>0:
        new_stroke_data = [image[0]]
        image=np.array(image)
        height, width = np.max(image[:, :2], axis=0) - np.min(image[:, :2], axis=0)
        t_dist = max(height, width) * thres
        for i in range(1, len(image) - 1):
            distance = eucliDist(image[i-1], image[i])
            if distance > t_dist:
                new_stroke_data.append(image[i])
    else:
        new_stroke_data=image
    return new_stroke_data
def normalize(image):
    Lt_all,up_all,uq_all=0,0,0
    for i in range(0,len(image)-1):
        pt,pt1,qt,qt1=image[i][0],image[i+1][0],image[i][1],image[i+1][1]
        Lt_now=math.pow((pt1-pt)**2+(qt1-qt)**2,0.5)
        up=Lt_now*(pt+pt1)
        up_all=up_all+up
        uq=Lt_now*(qt+qt1)
        uq_all=uq_all+uq
        #print(Lt_now)
        Lt_all=Lt_all+Lt_now
    if Lt_all==0:
      Lt_all=1
    up=0.5*(up_all/Lt_all)
    uq=0.5*(uq_all/Lt_all)
    zzz=0
    for i in range(0,len(image)-1):
        pt, pt1, qt, qt1 = image[i][0], image[i + 1][0], image[i][1], image[i + 1][1]
        Lt_now = math.pow((pt1 - pt) ** 2 + (qt1 - qt) ** 2, 0.5)
        zzz=zzz+Lt_now*((pt1-up)**2)+(pt1-up)*(pt-up)+(pt-up)**2
    if zzz==0:
      zzz=1
    standard=math.pow((zzz)/(3*Lt_all),0.5)
    for i in range(0,len(image)):
        pt, qt= image[i][0],  image[i][1]
        image[i]=[(pt-up)/standard,(qt-uq)/standard]
    return image
def normalize_trajectory(traj, thres=0.015):
    traj = removeRedundantPoints(traj, thres)
    traj = normalize(traj)
    return traj
def write2txt(dat_files,write2path='Couch_GB2_TXT'):
    jishu=0
    for i in dat_files:
        print(jishu)
        jishu=jishu+1
        name = i.split("/")[-1][:-4]
        file_path=os.path.join(write2path,name)
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
            print('正在创建文件夹')
        all_stroke,all_label,danbi=get_coor(i,class_num)
        if len(all_label)!=len(all_stroke):
            print('存在异常')
            break
        for data in range(len(all_stroke)):
            traj=all_stroke[data]
            traj=normalize_trajectory(traj)
            txtfile_name=name+'_'+str(data)+'.txt'
            txt_path=os.path.join(file_path,txtfile_name)
            f = open(txt_path, "w")
            f.write(str(traj))
            f.close()
def write2image(dat_files,write2path='Couch_GB2_IAMGE'):
    jishu=0
    for i in dat_files:
        print(jishu)
        jishu=jishu+1
        name = i.split("/")[-1][:-4]
        file_path=os.path.join(write2path,name)
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
            print('正在创建文件夹')
        all_stroke,all_label,danbi=get_coor(i,class_num)
        if len(all_label)!=len(all_stroke):
            print('存在异常')
            break
        #print(len(all_stroke))
        for data in range(len(danbi)):
            fig = plt.figure(figsize=(0.64, 0.64))
            for j in range(len(danbi[data])):
                imagefile_name=name+'_'+str(data)+'.png'
                image_path = os.path.join(file_path, imagefile_name)
                trajs = np.array(danbi[data][j])
                plt.axis('off')
                plt.plot(trajs[:, 0], trajs[:, 1],color='black')
            plt.savefig(image_path)
            plt.cla()
            plt.close("all")
dat_files,class_num=get_dat('SCUT-DATASET/Couch_GB2_195')
write2txt(dat_files)
write2image(dat_files)



































