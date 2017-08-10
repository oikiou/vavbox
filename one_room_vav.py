#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

# ## 说明
# 只有一个房间 一台VAV 一台AHU (之后考虑3个房间的3台VAV)
# 房间大小 8m*8m 高度3m 室内设计温度26摄氏度 日均负荷100w/m2 峰值负荷150w/m2 日均整体负荷6.4kw，设计负荷9.6kw
# 设备选用一台VAV 额定容量10kw 送风温度18摄氏度 温差8摄氏度
# VAV选型 Q=cm*dt, m=10/1.005/8=1.244[kg/s], rou=1.213, V=m/rou=1.025[m3/s]=3692[m3/h] 取整到4000[m3/h]
# VAV能力 Q=4000*1.213*8*1.005/3600=10.8kw
# 风管选型 假定风速5m/s 截面积A=4000/3600/5=0.222m2 半径r=0.266m 取整到r=0.3m, A=0.2826m2, v=3.93m/s
# 风管阻力系数 dp=SG2, S=3.125*10^-6[pa*h2/m6] 末端阻力系数 Se=1.875*10^-6[pa*h2/m6]
# VAV风阀特性 dp=G2/Cv2/k/p1, k=16000, cv全开时，20pa，cv半开时，80pa，cv下限时，减小inv
# 风机选型 额定压力 150pa 额定风量 4000
# 风机特性曲线 inv全开 (2000, 200; 4000, 150; 6000, 70) 三点确定曲线
# AHU控制逻辑 末端压力控制 定压15pa 变压力控制和变温度控制暂无

# ## 计算流程
# 前一时刻 VAV开度 INV开度
# 当前时刻 管路平衡 计算 送风量 末端压力
# 当前时刻 有限差分法 计算 室内负荷 计算 室内温度
# PID控制 末端压力和设定压力 控 INV开度 室内温度和设定温度 控 VAV开度
# 下一时刻

# ## 代码逻辑
# 说明
# 输入
# 有限差分法函数
# 风机特性曲线函数
# 管路平衡函数
# 二分法函数
# 分时函数
# PID函数
# 初始化
# 循环开始
# 计算流程
# 循环结束
# 输出

# input
# 外气参数
indoor_temp_set = 26
out_temp = [29.1, 28.9, 28.6, 28.4, 28.3, 28.7, 29.5, 30.5, 31.5, 32.3, 33.1, 33.8, 34.3, 34.6, 34.5, 34.0, 33.8,
            33.1, 32.2, 31.3, 30.7, 30.2, 29.8, 29.6]
# 日照得热(太阳辐射强度)(设计日)
sun_south = [0, 0, 0, 0, 0, 18, 50, 79, 134, 217, 273, 291, 273, 217, 134, 79, 50, 18, 0, 0, 0, 0, 0, 0]
sun_west = [0, 0, 0, 0, 0, 18, 50, 79, 102, 119, 130, 133, 336, 505, 615, 640, 558, 353, 0, 0, 0, 0, 0, 0]
sun_east = [0, 0, 0, 0, 0, 353, 558, 640, 615, 505, 336, 133, 130, 119, 102, 79, 50, 18, 0, 0, 0, 0, 0, 0]
sun_north = [0, 0, 0, 0, 0, 125, 148, 118, 102, 119, 130, 133, 130, 119, 102, 118, 148, 125, 0, 0, 0, 0, 0, 0]
sun_h = [0, 0, 0, 0, 0, 88, 276, 487, 681, 836, 933, 967, 933, 836, 681, 487, 276, 88, 0, 0, 0, 0, 0, 0]
# 内墙绝热，屋顶地板绝热
# 尺寸形状参数
out_wall_length = [8, 8, 0, 0]
out_window_length = [6, 6, 0, 0]
window_height = 2
wall_height = 3
ground_area = 64
# 物性参数
wall_lambda = 1.28  # 热传导率/导热系数[W/mK]
wall_rou = 2500  # 密度[kg/m3]
wall_c = 970  # 比热[k/kg/K]
wall_d = 0.2  # 厚度[m]
wall_alpha = 0.7  # 表面吸收率[-]
window_k = 5.2  # 热贯流率/传热系数[W/m2K]
window_tao = 0.7  # 透过率[-]
wall_h_inside = 9  # 内表面热传达率[W/m2K]
wall_h_outside = 23  # 外表面热传达率[W/m2K]
# 室内空气
air_c = 1.005
air_rou = 1.2
# 内扰
human_heat = 80  # [W/人]
human_count = 6
light_heat = 18  # [W/m2]
equipment_heat = 20  # [W/m2]
# 作息
schedule = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
light_schedule = [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 0.5, 0, 0]
# 差分法
wall_n = 10  # 墙体网格数
wall_dt = 60  # 时间间隔[s]
# PID控制
vav_box_01_pid_p = 0.02
vav_box_01_pid_i = 0.00001
vav_box_01_pid_d = 0
ahu_01_pid_p = 0.01
ahu_01_pid_i = 0.0001
ahu_01_pid_d = 0
# 配管特性
end_p_set_point = 15
supply_air_P_set_point = 80
duct_s1 = 0.000003125
duct_se1 = 0.000001875
vav_box_01_cv_k = 16000

# 输入信息预处理
# 面积计算
out_wall_area = [i * wall_height for i in out_wall_length]
out_window_area = [i * window_height for i in out_window_length]
out_wall_area = [out_wall_area[i] - out_window_area[i] for i in range(4)]
# 差分法计算
wall_dx = wall_d / wall_n  # 网格大小
wall_Fo = wall_lambda * wall_dt / wall_rou / wall_c / wall_dx / wall_dx  # 网格傅里叶数
wall_Bi_inside = 2 * wall_h_inside * wall_dx / wall_lambda  # 墙内边界网格毕渥数
wall_Bi_outside = 2 * wall_h_outside * wall_dx / wall_lambda  # 墙外边界网格毕渥数


# 有限差分法 计算墙体非定常传热
def k_to_k1_side(tk, t1k, tf, fo, bi):
    return tk * (1 - fo * bi - 2 * fo) + 2 * t1k * fo + fo * bi * tf


def k_to_k1_mid(tk, t1k, t2k, fo=wall_Fo):
    return tk + fo * (t1k - tk) + fo * (t2k - tk)


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


# 利用风机特性曲线，求压力
def g_to_p_fan(g, inv, performance):
    g1 = [1, g, g*g]
    return (g1 * performance * inv).tolist()[0][0]

# 特性曲线图
x_pump1 = np.linspace(0, 6000, 50)
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


# 风机特性算压力
def f_gp_fan(g):
    return g_to_p_fan(g, ahu_01_fan_inv, fan_1_performance)


# 配管特性算压力
def f_gp_duct(g):
    result = cal_g1_to_p0(g, duct_s1, duct_se1, vav_box_01_cv, vav_box_01_cv_k)
    return result


# 二分法求两个函数的交点
def two_solve(max_x, min_x, f1, f2, e):
    x = 1/2 * max_x + 1/2 * min_x
    y1 = f1(x)
    y2 = f2(x)
    if math.fabs(y1 - y2) < e:
        return x, y1, y2
    else:
        if y1 < y2:
            max_x = x
        else:
            min_x = x
        return two_solve(max_x, min_x, f1, f2, e)


# 分时
def one_in_sixty(one, begin=0):
    sixty = np.linspace(begin, one[0], 60)
    for i in range(len(one)-1):
        temp = np.linspace(one[i], one[i+1], 60)
        sixty = np.concatenate((sixty, temp), axis=0)
    return sixty


def one_to_sixty_plus(one, begin=0):
    sixty = []
    for i in range(len(one)):
        end = one[i] * 2 / 60 - begin
        temp = np.linspace(begin, end, 60)
        sixty = np.concatenate((sixty, temp), axis=0)
        begin = end
    return sixty

# 进行分时
# temp hour in min
out_temp_min = one_in_sixty(out_temp, out_temp[0])
# sun hour to min
sun_east_min = one_in_sixty(sun_east)
sun_south_min = one_in_sixty(sun_south)
sun_west_min = one_in_sixty(sun_west)
sun_north_min = one_in_sixty(sun_north)
sun_h_min = one_in_sixty(sun_h)
# schedule
schedule_min = one_in_sixty(schedule)
light_schedule_min = one_in_sixty(light_schedule)


# pid control
# init e0,es, in ct0 ta0, out ct1
def pid_control(sche, target, set_point, control0, p, i, d, e0, es, control_max=1, control_min=0, tf=1):
    if sche == 0:
        return 0, 0, 0
    else:
        e = target - set_point
        de = e - e0
        if de * e <= 0:
            de = 0
        es += e
        control = max(min(control0 + tf * (e * p + es * i + de * d), control_max), control_min)
        return control, e, es

# 初始化 init
indoor_temp = 26
supply_air_temp = 18
wall_t0 = [np.linspace(indoor_temp, out_temp_min[0], wall_n)] * 4
vav_box_01_cv = 0
vav_box_01_pid_e0 = 0
vav_box_01_pid_es = 0
ahu_01_fan_inv = 0
ahu_01_fan_inv_pid_e0 = 0
ahu_01_fan_inv_pid_es = 0

# 循环开始
# ## 计算流程
# 前一时刻 VAV开度 INV开度
# 当前时刻 管路平衡 计算 送风量 末端压力 制冷能力
# 当前时刻 有限差分法 计算 室内负荷 计算 室内温度
# PID控制 末端压力和设定压力 控 INV开度 室内温度和设定温度 控 VAV开度
# 下一时刻
for step in range(1440):
    # 管路平衡
    if vav_box_01_cv == 0 or schedule_min[step] == 0:
        supply_air_G = 0
        supply_air_P = 0
        vav_box_01_dp = 0
        end_dp = 0
    else:
        supply_air_G, supply_air_P, p_check = two_solve(6000, 0, f_gp_fan, f_gp_duct, 0.1)
        vav_box_01_dp = cal_g1_to_p0(supply_air_G, duct_s1, duct_se1, vav_box_01_cv, vav_box_01_cv_k)
        end_dp = cal_p_end(supply_air_G, vav_box_01_cv, vav_box_01_cv_k, vav_box_01_dp)

    # 制冷能力 W
    vav_box_01_capacity = (indoor_temp - supply_air_temp) * supply_air_G * air_c * air_rou / 3.6 * schedule_min[step]

    # 负荷计算 W
    # sun
    sun_min = [sun_east_min[step], sun_south_min[step],
               sun_west_min[step], sun_north_min[step]]
    load_window_tou = [out_window_area[i] * sun_min[i] * window_tao for i in range(4)]
    load_window_den = (out_temp_min[step] - indoor_temp) * window_k * sum(out_window_area)
    load_window = sum(load_window_tou) + load_window_den

    # wall
    wall_sat = [out_temp_min[step] + sun_min[i] * wall_alpha / wall_h_outside for i in range(4)]
    wall_t1 = [[0] * wall_n] * 4
    for orientation in range(4):
        wall_t1[orientation][0] = k_to_k1_side(wall_t0[orientation][0], wall_t0[orientation][1],
                                               indoor_temp, wall_Fo, wall_Bi_inside)
        wall_t1[orientation][wall_n-1] = k_to_k1_side(wall_t0[orientation][wall_n-1], wall_t0[orientation][wall_n-2],
                                                      wall_sat[orientation], wall_Fo, wall_Bi_outside)
        for i in range(wall_n - 2):
            wall_t1[orientation][i+1] = k_to_k1_mid(wall_t0[orientation][i+1], wall_t0[orientation][i],
                                                    wall_t0[orientation][i+2])
        wall_t0[orientation] = wall_t1[orientation]
    load_wall = sum([wall_h_inside * (wall_t1[orientation][0] - indoor_temp) * out_wall_area[orientation]
                     for orientation in range(4)])

    # 内扰
    load_human = human_count * human_heat * schedule_min[step]
    load_light = light_heat * light_schedule_min[step] * ground_area
    load_equipment = equipment_heat * schedule_min[step] * ground_area

    # load_sum
    load_sum = load_wall + load_window + load_human + load_light + load_equipment

    # indoor_temp
    delta_load = load_sum - vav_box_01_capacity
    # 热容只考虑空气，不考虑房间内其他家具
    delta_temp = delta_load / air_rou / air_c / ground_area / wall_height / 1000 * 60
    indoor_temp = indoor_temp + delta_temp

    # PID控制
    # 风阀控制
    [vav_box_01_cv, vav_box_01_pid_e0, vav_box_01_pid_es] = pid_control(schedule_min[step], indoor_temp,
                                                                        indoor_temp_set, vav_box_01_cv,
                                                                        vav_box_01_pid_p, vav_box_01_pid_i,
                                                                        vav_box_01_pid_d, vav_box_01_pid_e0,
                                                                        vav_box_01_pid_es, control_min=0.1)
    '''
    # INV控制末端压力 ##
    [ahu_01_fan_inv, ahu_01_fan_inv_pid_e0, ahu_01_fan_inv_pid_es] = pid_control(schedule_min[step], end_dp,
                                                                                 end_p_set_point, ahu_01_fan_inv,
                                                                                 ahu_01_pid_p, ahu_01_pid_i,
                                                                                 ahu_01_pid_d, ahu_01_fan_inv_pid_e0,
                                                                                 ahu_01_fan_inv_pid_es, control_min=0.3,
                                                                                 tf=-1)
  
    # INV控制出口压力 阀门全开，压力设定++， 阀门最小，压力设定--
    if vav_box_01_cv > 0.9:
        supply_air_P_set_point += 1
    elif vav_box_01_cv < 0.5:
        supply_air_P_set_point -= 1
    else:
        supply_air_P_set_point = 80
    '''

    # INV控制出口压力
    [ahu_01_fan_inv, ahu_01_fan_inv_pid_e0, ahu_01_fan_inv_pid_es] = pid_control(schedule_min[step], supply_air_P,
                                                                                 supply_air_P_set_point, ahu_01_fan_inv,
                                                                                 ahu_01_pid_p, ahu_01_pid_i,
                                                                                 ahu_01_pid_d, ahu_01_fan_inv_pid_e0,
                                                                                 ahu_01_fan_inv_pid_es, control_min=0.3,
                                                                                 tf=-1)

    # INV控制流量
    print(load_wall, sum(load_window_tou), load_window_den, load_human, load_light, load_equipment,
          load_sum, indoor_temp, out_temp_min[step], vav_box_01_cv, ahu_01_fan_inv, supply_air_P)
    # 循环结束



