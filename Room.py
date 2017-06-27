#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import numpy
import random

# load calculation for vav system

seed = 10021

outtemp = [22,23,23,25,26,28,28,28,29,29,30,31,31,32,32,31,30,29,29,29,28,26,24,20] #外气温度
month = 7  # 月
day = 23  # 日
time = 11  # 时间

class Room:
    indoortempset = 26 #室内设定温度
    height = 3 #房间高度
    windowsratio = 0.7 #窗墙比
    wall_k = 0.9 #墙体K值
    window_k = 2.4 #窗K值
    load_delay = [0.1,0.18,0.16,0.13,0.1,0.08,0.06,0.05,0.04,0.03,0.025,0.02,0.015,0.01] #冷负荷系数
    air_density = 1.2 #空气密度
    air_people_need = 30 #人均最低新风量
    safecase = 1.2 #安全系数
    people_per_square = 15 #人均占地面积
    window_tou_ratio = 0.6 #窗体透过率
    schedule_people = [0.05,0,0,0,0,0,0,0.05,0.2,0.3,0.5,0.5,0.6,0.6,0.7,0.7,0.7,0.6,0.5,0.4,0.3,0.3,0.2,0.1] #人员作息
    schedule_equipment = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0] #设备作息
    city = 'Tokyo' #城市
    longitude = 139.46 #经度
    latitude = 35.42 #纬度
    sun_stand = 1340 #太阳常数
    cloud_ratio = 0.6 #云层穿透率（平均值）

    def __init__(self,out_wall_length,in_wall_length,area,angel):
        self.out_wall_length = out_wall_length
        self.in_wall_length = in_wall_length
        self.area = area
        self.angel = angel

    def load_cal(self,out,mouth,day,time):
        #area
        self.volumn = self.area * self.height #房间体积
        self.windowarea = self.out_wall_length * self.height * self.windowsratio #外窗面积
        self.out_wall_area = self.out_wall_length * self.height * (1-self.windowsratio) #外墙面积
        self.in_wall_area = self.in_wall_length * self.height #内墙面积
        self.people = round(self.area / self.people_per_square,0) #人数

        #load 24时间
        self.load_outwall0 = max(out[time] - self.indoortempset, 0) * self.wall_k * self.out_wall_area #外墙得热
        self.load_outwall = 0 #外墙负荷
        for i in range(len(out)):
            temp = 0
            for j in range(len(self.load_delay)):
                if i >= j:
                    temp += self.load_outwall0[(i-j)] * self.load_delay[j]
                else:
                    break
            self.load_outwall = temp
        self.load_inwall = [3 * self.in_wall_area * self.wall_k * p for p in self.schedule_equipment] #内墙负荷
        self.load_indoor = [self.people * 60 * p for p in self.schedule_people] #内部发热
        self.load_freshair = [self.people * self.air_people_need * max(ot - self.indoortempset, 0) *
                              self.air_density * self.safecase for ot in out] #新风负荷
        self.load_window = [max(ot - self.indoortempset, 0) * self.window_k * self.windowarea for ot in out] #窗传热负荷
        self.sun_cal(mouth,day,time)
        #self.load_sun =

    def sun_cal(self,month,day,time):
        self.day_N = month * 30 + day  # 一年的第几天
        self.sin_declination = 0.39795 * math.cos(0.98563 * (self.day_N - 173))  # 太阳赤纬
        self.time_angel = numpy.linspace(-180,165,24)  # 时角
        self.h = [math.asin(math.sin(self.latitude) * self.sin_declination + math.cos(self.latitude) *
                           math.cos(math.asin(self.sin_declination)) * math.cos(t)) for t in self.time_angel] #太阳高度角
        self.A_sin =[]
        for i in range(24):
            self.A_sin.append(math.cos(math.asin(self.sin_declination)) * math.sin(self.time_angel[i]) / math.cos(self.h[i]))
        self.A_cos = [(math.sin(sh) * math.sin(self.latitude) - self.sin_declination) /
                      (math.cos(math.asin(sh)) * math.cos(self.latitude)) for sh in self.h]

    def print(self):
        pass

r1 = Room(16,16,64,-45)
r2 = Room(8,24,64,0)
r3 = Room(16,16,64,45)
r = [r1,r2,r3]

r1.load_cal(outtemp,month,day,time)

'''
print(r1.load_outwall0)
print(r1.load_outwall)
print(r1.load_freshair)
print(r1.load_inwall)
print(r1.load_window)
print(r1.time_angel)
print(r1.sin_declination)
print(r1.h)
print(r1.A_sin)
print(r1.A_cos)
'''