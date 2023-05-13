'''
This script is used to check 'online handwrite words'.

usage: python check.py {DirectoryPtah}
'''
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
def load_traj_2016(filename):
    with open(filename, "r") as f:
        txt = []
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split()
            line = list(map(float, line))
            txt.append(line)
        txt.pop(0)
    return np.array(txt)
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

def draw_traj11(trajs,path):
    trajs = removeRedundantPoints(trajs)
    trajs=np.array(trajs)
    fig = plt.figure(figsize=(1.28, 1.28))
    plt.axis('off')
    plt.plot(trajs[:,0], trajs[:, 1], marker = None,color='black')
    fig.savefig(path)
    plt.cla()
    plt.close("all")
    

def traj_image_save(path1,path2):
    trainnames = os.listdir(path1)
    if os.path.exists(path2):
        print('该文件已存在')
    else:
        os.makedirs(path2)
    jishu=0
    for filename in trainnames:
        print(jishu)
        jishu=jishu+1
        filepath=os.path.join(path1, filename)
        traj=load_traj_2016(filepath)
        imagepath = os.path.join(path2, str(filename)[:-4]+'.png')
        draw_traj11(traj,imagepath)
        

traj_image_save('IAHCC-UCAS2016-TXT','image_test_2016')
