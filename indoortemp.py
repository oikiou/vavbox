#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

# 模拟实际的房间控制
# step:1min
# length:7days
# load:solar wall window cell human light equipment sukima
# 单个房间 自然室温变化 阀门pid控室温 管路阻力变化 风机pid控压力(+变压力+变温度) 风量变化 下一时刻的自然室温计算
# 多个房间

# input 外界参数
indoor_temp_set = 26
out_temp = [22, 23, 23, 25, 26, 28, 28, 28, 29, 29, 30, 31, 31, 32, 32, 31, 30, 29, 29, 29, 28, 26, 24, 20]
side_room_temp = [min(29, i) for i in out_temp]
out_wall_length = [8, 8, 0, 0]
in_wall_length = [0, 0, 8, 8]

out_window_length = [6, 6, 0, 0]
window_height = 2
wall_height = 3
ground_area = 64
k_wall = 1.2
k_window = 3.4
k_in_wall = 1.8
t_window = 0.5
t_wall = 0.3
c_air = 1.005
rou_air = 1.2
schedule = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
wall_delay = [0.1, 0.18, 0.16, 0.13, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015,
              0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
wall_delay_indoor = [0.15, 0.24, 0.20, 0.18, 0.10, 0.07, 0.04, 0.02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

vav_01_pid_p = 0.02
vav_01_pid_i = 0.00001
vav_01_pid_d = 0
fan_inv_p = 0.001
fan_inv_i = 0.0001
fan_inv_d = 0

out_wall_area = [i * wall_height for i in out_wall_length]
out_window_area = [i * window_height for i in out_window_length]
out_wall_area = [out_wall_area[i] - out_window_area[i] for i in range(4)]
in_wall_area = [i * wall_height for i in in_wall_length]
roof_area = ground_area
in_wall_area.append(roof_area)

sun_radit_east = [0, 0, 0, 0, 0, 0, 120, 240, 500, 480, 380, 200, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sun_radit_east = [0, 0, 0, 0, 0, 0, 70, 130, 260, 250, 200, 110, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sun_radit_south = [0, 0, 0, 0, 0, 0, 60, 180, 250, 400, 500, 680, 500, 400, 250, 180, 60, 0, 0, 0, 0, 0, 0, 0]
sun_radit_west = [0] * 24
sun_radit_north = [0] * 24

human_rate = 80
human_count = 6
human_schedule = schedule

light_rate = 15
light_schedule = [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.7, 0.8, 0.8, 0, 0, 0, 0]

equipment_rate = 20
equipment_schedule = schedule

vav_schedule = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
# vav_schedule = [0] * 24  # for natural indoor air temp


# ## 风机特性
# 任意三点求二次曲线
def three_to_performance(x1, y1, x2, y2, x3, y3):
    x = np.mat([[1,x1,x1*x1],[1,x2,x2*x2],[1,x3,x3*x3]])
    y = np.mat([y1,y2,y3])
    return np.reshape(x.I * y.T,[-1,1])


# 特定频率下的三个工况点求水泵、风机特性曲线
# pump fan performance characteristics
def performance_characteristics(inv, x1, y1, x2, y2, x3, y3):
    return three_to_performance(x1, y1/inv, x2, y2/inv, x3, y3/inv)

# 1号风机的特性曲线
fan_1_performance = performance_characteristics(0.75, 0, 200, 3000, 150, 6000, 0)


# 利用风机特性曲线，求压力
def g_to_p_fan(g, inv, performance):
    g1 = [1, g, g*g]
    return (g1 * performance * inv).tolist()[0][0]

# 特性曲线图
x_pump1 = np.linspace(0,6000,50)
y_pump1 = []
for i in range(50):
    y_pump1.append(g_to_p_fan(x_pump1[i], 0.75, fan_1_performance))
# plt.plot(x,y)
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

# duct balance
s1 = 0.000005
se1 = 0.000003
cv_k = 5000


# 风机特性算压力
def f_gp_fan(g):
    return g_to_p_fan(g, fan_inv, fan_1_performance)


# 配管特性算压力
def f_gp_duct(g):
    result = cal_g1_to_p0(g, s1, se1, vav_cv, cv_k)
    return result


# 二分法
def two_solve(maxx, minx, f1, f2, e):
    x0 = 1/2 * maxx + 1/2 * minx
    y1 = f1(x0)
    y2 = f2(x0)
    # print(x0,y1,y2)
    if math.fabs(y1 - y2) < e:
        return x0,y1,y2
    else:
        if y1 < y2:
            maxx = x0
        else:
            minx = x0
        return two_solve(maxx, minx, f1, f2, e)

'''
def duct_balance(cv, inv, s, se, k):
    g_max = 6000
    g_min = 0
    g0 = 0.5 * (g_max + g_min)
    for i in range(100):
        # p = g_to_p
        # p = p1 + s1 * g * g
        # p1 = se1 * g * g + g * g / cv / cv / cv_k / p1
        p1_a = 1
        p1_b = - se * g0 * g0
        p1_c = - g0 * g0 / cv / cv / k
        p1 = cal_p1(p1_a, p1_b, p1_c)
        p_duct = p1 + s * g0 * g0
        error_p = p_duct - g_to_p(g0,inv/100,fan_1_performance)
        if error_p < 0.01 and error_p > -0.01:
            break
        if error_p > 0:
            g1 = 0.5 * (g_min + g0)
            g_max = g0
        else:
            g1 = 0.5 * (g_max + g0)
            g_min = g0
        g0 = g1
    dp = g0 * g0 / cv / cv / k / p1
    p_end = p1 - dp
    return g0, p_duct, dp, p_end
'''


# pre
def one_to_sixty(one):
    sixty = []
    for i in range(len(one)):
        temp = [one[i]/60] * 60
        sixty = np.concatenate((sixty, temp), axis=0)
    return sixty


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

# temp hour in min
out_temp_min = one_in_sixty(out_temp, 20)
side_room_temp_min = one_in_sixty(side_room_temp, 24)
# delay hour to min
# wall_delay_min = one_to_sixty(wall_delay)
# wall_delay_indoor_min = one_to_sixty(wall_delay_indoor)
wall_delay_min = one_to_sixty_plus(wall_delay)
# wall_delay_min1 = one_to_sixty(wall_delay)
# plt.plot(wall_delay_min)
# plt.plot(wall_delay_min1)
# plt.show()
wall_delay_indoor_min = one_to_sixty_plus(wall_delay_indoor)
# sun_radit hour to min
sun_radit_east_min = one_in_sixty(sun_radit_east)
sun_radit_south_min = one_in_sixty(sun_radit_south)
sun_radit_west_min = one_in_sixty(sun_radit_west)
sun_radit_north_min = one_in_sixty(sun_radit_north)
# schedule
schedule_min = one_in_sixty(schedule)
light_schedule_min = one_in_sixty(light_schedule)
vav_schedule_min = one_in_sixty(vav_schedule)


# pid control
# init e0,es, in ct0 ta0, out ct1
def pid_control(schedule, target, setpoint, control0, p, i, d, e0, es, control_max=1, control_min=0, tf=1):
    if schedule == 0:
        return 0, 0, 0
    else:
        e = target - setpoint
        de = e - e0
        if de * e <= 0:
            de = 0
        es += e
        control = max(min(control0 + tf * (e * p + es * i + de * d), control_max), control_min)
        return control, e, es

# init
indoor_temp = 26
load_stun = [0] * 1440
load_stun_indoor = [0] * 1440
supply_air_G = 2500
supply_air_temp = 18

vav_cv = 0
vav_01_pid_e0 = 0
vav_01_pid_es = 0

p_end = 0
p_end_setpoint = 30
fan_inv = 0
fan_inv_pid_e0 = 0
fan_inv_pid_es = 0

# cal indoor_temp
for step in range(1440):
    # ## load
    # sun_load
    sun_radit_min = [sun_radit_east_min[step], sun_radit_south_min[step],
                     sun_radit_west_min[step], sun_radit_north_min[step]]
    load_sun_window = [out_window_area[i] * sun_radit_min[i] * t_window for i in range(4)]
    load_sun = sum(load_sun_window)

    # wall_load
    load_sun_wall_single = [out_wall_area[i] * sun_radit_min[i] * t_wall for i in range(4)]
    load_sun_wall = sum(load_sun_wall_single)
    heat_got = (out_temp_min[step] - indoor_temp) * k_wall * sum(out_wall_area) + load_sun_wall  # W
    # print(out_temp_min[step],k_wall,sum(out_wall_area))
    # print(heat_got)
    load_stun = [load_stun[i] + heat_got * wall_delay_min[i] for i in range(1440)]  # J/s
    load_wall = load_stun[0]
    load_stun = load_stun[1:]
    load_stun.append(0)
    # print(load_wall)

    # window_load
    load_window = (out_temp_min[step] - indoor_temp) * k_window * sum(out_window_area)

    # indoor_wall_load
    heat_got_indoor = (side_room_temp_min[step] - indoor_temp) * k_in_wall * sum(in_wall_area)
    load_stun_indoor = [load_stun_indoor[i] + heat_got_indoor * wall_delay_indoor_min[i] for i in range(1440)]
    load_wall_indoor = load_stun_indoor[0]
    load_stun_indoor = load_stun_indoor[1:]
    load_stun_indoor.append(0)

    # human_load
    load_human = human_count * human_rate * schedule_min[step]

    # light_load
    load_light = light_rate * ground_area * light_schedule_min[step]

    # equipment_load
    load_equipment = equipment_rate * ground_area * schedule_min[step]

    # load_sum
    load_sum = load_wall + load_sun + load_window + load_wall_indoor + load_human + load_equipment + load_light

    # print(load_wall,load_sun,load_window,load_wall_indoor,load_human,load_equipment,load_light,load_sum)

    # ## room_air_temp
    # vav (no control) control later
    vav_capicity = (indoor_temp - supply_air_temp) * supply_air_G * c_air * rou_air / 3.6 * vav_schedule_min[step]
    delta_temp = (load_sum - vav_capicity) / rou_air / c_air / ground_area / wall_height / 1000 * 60
    indoor_temp = indoor_temp + delta_temp
    # print(load_wall,load_sun,load_window,load_wall_indoor,load_human,load_equipment,load_light,load_sum,
          # out_temp_min[step], delta_temp, indoor_temp)

    # ## vav valve pid
    [vav_cv, vav_01_pid_e0, vav_01_pid_es] = pid_control(
        vav_schedule_min[step], indoor_temp, indoor_temp_set, vav_cv, vav_01_pid_p,
        vav_01_pid_i, vav_01_pid_d, vav_01_pid_e0, vav_01_pid_es)
    '''
    # error
    pid_e = indoor_temp - indoor_temp_set
    pid_de = pid_e - pid_e0
    # pid_d
    if pid_de * pid_e >= 0:
        pid_ds = -pid_de
    else:
        pid_ds = 0
    # restart i d
    if vav_schedule_min[step] == 0:
        pid_es = 0
        pid_ds = 0
    # pid_i
    pid_es += pid_e
    # valve
    vav_cv1 = min(max(vav_cv0 + pid_e * pid_p + pid_es * pid_i - pid_ds * pid_d,0),1)
    # G
    if vav_cv0 == 0:
        supply_air_G = 0
    else:
        supply_air_G = 4000 * vav_cv1
    # last_init
    vav_cv0 = vav_cv1
    pid_e0 = pid_e
    #print(indoor_temp)

    # fan_inv_pid
    # setpoint p_end = 30
    p_end_setpoint = 30
    p_error = p_end - p_end_setpoint
    p_derror = p_error - p_error0
    if p_derror * p_error >= 0:
        p_ds = - p_derror
    else:
        p_ds = 0
    # vav--ahu--schedule
    if vav_schedule_min[step] == 0:
        p_es = 0
        p_ds = 0
    p_es += p_error
    fan_inv = min(max(fan_inv - p_error * fan_inv_p - p_es * fan_inv_i + p_ds * fan_inv_d,10),100)
    p_error0 = p_error
    '''

    # fan_inv_pid
    # pid 控制策略有问题
    [fan_inv, fan_inv_pid_e0, fan_inv_pid_es] = pid_control(
        vav_schedule_min[step], p_end, p_end_setpoint, fan_inv, fan_inv_p, fan_inv_i, fan_inv_d,
        fan_inv_pid_e0, fan_inv_pid_es, control_max=1, control_min=0, tf=-1)

    # ## duct pressure balance
    # cv = g / √pδp /cv_k

    if vav_cv == 0 or vav_schedule_min[step] == 0:
        supply_air_G = 0
        p_duct = 0
        p_vav = 0
        p_end = 0
    else:
        # [g0, p_duct, p_vav, p_end] = duct_balance(vav_cv, fan_inv, s1, se1, cv_k)
        supply_air_G, supply_air_p, p_check = two_solve(6000, 0, f_gp_fan, f_gp_duct, 0.1)
        p_duct = supply_air_p
        p_vav = cal_g1_to_p0(supply_air_G, s1, se1, vav_cv, cv_k)
        p_end = cal_p_end(supply_air_G, vav_cv, cv_k, p_vav)
    # print(supply_air_G)
    # print(g0,p_duct,p_vav,p_end)
    # supply_air_G = g0

    print(p_end,indoor_temp,supply_air_G,fan_inv,vav_cv)

    # print(indoor_temp)

    # ## TO DO LIST
    # room *3 邻室传热 + 风管平衡要解多元方程组
    # p reset 变压力控制
    # t reset 送风温度重置 (重要！！！ei)
    # lim sennsei hiteijyou 林先生的非定常计算法
    # temp without vav 自然室温计算 放在excle
    # 1分钟的时间间隔 室温来不及变化 非定常 非均一 怎么考虑
    # 把房间 vav ahu 风管 做成class 面向对象
