#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import numpy as np
#import matplotlib.pyplot as plt
#import tensorflow as tf
import random

# ## 用BPNN做风管平衡
# 三个房间 三个VAV 三个出风口 六段风管 一个AHU 一个风机
# VAV 作为阀门 阻力dp和阀门开度cv之间的关系 dp = g2/cv2/k/p1
# 出风口和风管 阻力dp和管道阻力系数的关系 dp = sg2
# 常量 管路特性参数 s1 s2 s3 se1 se2 se3 k1 k2 k3
# 变量 cv1 cv2 cv3 inv (random)
# 生成数据所用假定值 g1 (random)
# 计算流程 g1 - p1 - g2 - p3 - g3 - g - p - inv
# 目标值 inv
# 生成学习集 (cv1, cv2, cv3, inv), (g1, g2, g3)

# ## 神经网络
# 1隐层 4-5-3
# 输入变量 cv1 cv2 cv3 inv
# 输出变量 g1 g2 g3

# ## 管路选型
# 房间大小相同，负荷相近，VAV选用同一种设备，K = 16000
# 出风口选型相同，SE = 1.875 * 10 ^ -6
# 管长，管径，管道粗糙度默认相同， S = 3.125 * 10 ^ -6

k = 16000
se = 1.9 / 3600 / 3600
s1 = 2.2 / 3600 / 3600
s2 = 2.2 / 3600 / 3600
s3 = 2.2 / 3600 / 3600
s4 = 5.8 / 3600 / 3600
s5 = 1.3 / 3600 / 3600
s6 = 2.1 / 3600 / 3600


# ## 风机特性
# 任意三点求二次曲线
def three_point_to_performance(x1, y1, x2, y2, x3, y3):
    x = np.mat([[1, x1, x1*x1], [1, x2, x2*x2], [1, x3, x3*x3]])
    y = np.mat([y1, y2, y3])
    return np.reshape(x.I * y.T, [-1, 1])


# 特定频率下的三个工况点求水泵、风机特性曲线
# pump fan performance characteristics
def performance_characteristics(inv, x1, y1, x2, y2, x3, y3):
    return three_point_to_performance(x1, y1/inv, x2, y2/inv, x3, y3/inv)

# 1号风机的特性曲线(2000, 200; 4000, 150; 6000, 70)
fan_1_performance = performance_characteristics(1, 2000, 200, 4000, 150, 6000, 70)
# 2号风机的特性曲线
fan_2_performance = performance_characteristics(1, 6000, 600, 12000, 400, 18000, 210)
# 3号风机的特性曲线
fan_3_performance = performance_characteristics(2, 6000, 170, 12000, 150, 18000, 80)

# 利用风机特性曲线，求压力
def g_to_p_fan(g, inv, performance):
    g1 = [1, g, g*g]
    return (g1 * performance * inv).tolist()[0][0]

# 特性曲线图
x_pump1 = np.linspace(0, 18000, 50)
y_pump1 = []
for i in range(50):
    y_pump1.append(g_to_p_fan(x_pump1[i], 1, fan_1_performance))
# plt.plot(x_pump1,y_pump1)
# plt.show()


# ## 管路平衡
# 流量算VAV前压力
def cal_g1_to_p1(g, se, cv, k):
    return 0.5 * (se * g * g + math.sqrt(se * g * g * se * g * g + 4 * g * g / cv / cv / k))


# 算末端压力
def cal_p_end(g, cv, k, p1):
    return p1 - g * g / cv / cv / k / p1


# 流量算支管压差
def cal_g1_to_p0(g, s, se, cv, k):
    return cal_g1_to_p1(g, se, cv, k) + s * g * g


# 压差算支管流量
def cal_p0_to_g2(p0, s, se, cv, k):
    return math.sqrt((2 * p0 * s + se * p0 + 1 / cv / cv / k - math.sqrt((2 * p0 * s + se * p0 + 1 / cv / cv / k) *
         (2 * p0 * s + se * p0 + 1 / cv / cv / k) - 4 * (s * s + s * se) * p0 * p0))/(2 * (s * s + s * se)))

for i in range(60000):
    # 随机取初值
    cv1 = random.random()
    cv2 = random.random()
    cv3 = random.random()

    # 假定一个g1
    g1 = random.random() * 4000
    #g1 = cv1 * 6000 + (random.random() - 0.5) * 600

    # 计算流程 g1 - p1 - p2 - g2 - p3 - g3 - g - p
    # g1 - p1
    p1 = cal_g1_to_p1(g1, se, cv1, k)

    # p1 - p2
    p2 = p1 + s1 * g1 * g1 + s4 * g1 * g1

    # p2 - g2
    g2 = cal_p0_to_g2(p2, s2, se, cv2, k)

    # g2 - p3
    p3 = p2 + s5 * (g1 + g2) * (g1 + g2)

    # p3 - g3
    g3 = cal_p0_to_g2(p3, s3, se, cv3, k)

    if g2 > 4000 or g3 > 4000:
        continue

    # g3 - p
    p = p3 + s6 * (g1 + g2 + g3) * (g1 + g2 + g3)

    # p - inv
    pi1 = g_to_p_fan((g1 + g2 + g3), 1, fan_3_performance)
    inv = p / pi1

    if inv > 1 or inv < 0.2:
        continue

    drop_p = random.random()
    if inv * inv < drop_p:
        continue

    print(cv1, cv2, cv3, g1, g2, g3, g1 + g2 + g3, p, inv)









