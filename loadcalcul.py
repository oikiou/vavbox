#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import numpy
import random

# load calculation for vav system

seed = 10021
'''
class room:

    def __init__(self,east=0,south=0,west=0,north=0,eastin=0,southin=0,westin=0,northin=0,height=3,windowrate=0.7):
        self.east = east
        self.south = south
        self.west = west
        self.north = north
        self.eastin = eastin
        self.southin = southin
        self.westin = westin
        self.northin = northin
        self.height = height
        self.windowrate = windowrate
        self.calculation()

    def wallarea(self,wall):
        return ([x * self.height for x in wall])

    def calculation(self):
        self.wall = [self.east,self.south,self.north,self.west,self.eastin,self.southin,self.northin,self.westin]
        self.wallout = [self.east,self.south,self.west,self.north]
        self.wallin = [self.eastin,self.southin,self.westin,self.northin]
        self.walloutarea = self.wallarea(self.wallout)
        self.wallinarea = self.wallarea(self.wallin)
        self.windowarea = [x * self.windowrate for x in self.walloutarea]
        self.wall.sort()
        self.area = self.wall[-1] * self.wall[-2]
        self.volumn = self.area * self.height

room1 = [room(8,8,0,0,0,0,8,8),room(0,8,0,0,8,0,8,8),room(0,8,8,0,8,0,0,8)]
print('外墙面积',[(room.walloutarea) for room in room1])
print('外窗面积',[(room.windowarea) for room in room1])

class room:

    def __init__(self):
        self.calculation()

    def wallarea(self,wall):
        return ([x * self.height for x in wall])

    def calculation(self):
        self.wall = [self.east,self.south,self.north,self.west,self.eastin,self.southin,self.northin,self.westin]
        self.wallout = [self.east,self.south,self.west,self.north]
        self.wallin = [self.eastin,self.southin,self.westin,self.northin]
        self.walloutarea = self.wallarea(self.wallout)
        self.wallinarea = self.wallarea(self.wallin)
        self.windowarea = [x * self.windowrate for x in self.walloutarea]
        self.wall.sort()
        self.area = self.wall[-1] * self.wall[-2]
        self.volumn = self.area * self.height


room1.east = 8
room1.south = 8
room1.west = 0
room1.north = 0
room1.eastin = 0
room1.southin = 0
room1.westin = 8
room1.northin = 8
room1.height = 3
room1.windowrate = 0.7

print(room1.windowarea)


room_1_wall_east_length = 8;
room_1_wall_east_height = 3;
room_1_wall_south_length = 8;
room_1_wall_south_height = 3;
room_1_wall_south_length = 8;
room_1_wall_south_height = 3;
'''


outtemp = [22,23,23,25,26,28,28,28,29,29,30,31,31,32,32,31,30,29,29,29,28,26,24,20] #外气温度
month = 7  # 月
day = 23  # 日

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

    def load_cal(self,out,mouth,day):
        #area
        self.volumn = self.area * self.height #房间体积
        self.windowarea = self.out_wall_length * self.height * self.windowsratio #外窗面积
        self.out_wall_area = self.out_wall_length * self.height * (1-self.windowsratio) #外墙面积
        self.in_wall_area = self.in_wall_length * self.height #内墙面积
        self.people = round(self.area / self.people_per_square,0) #人数

        #load 24时间
        self.load_outwall0 = [max(ot - self.indoortempset, 0) * self.wall_k * self.out_wall_area for ot in out] #外墙得热
        self.load_outwall = [] #外墙负荷
        for i in range(len(out)):
            temp = 0
            for j in range(len(self.load_delay)):
                if i >= j:
                    temp += self.load_outwall0[(i-j)] * self.load_delay[j]
                else:
                    break
            self.load_outwall.append(temp)
        self.load_inwall = [3 * self.in_wall_area * self.wall_k * p for p in self.schedule_equipment] #内墙负荷
        self.load_indoor = [self.people * 60 * p for p in self.schedule_people] #内部发热
        self.load_freshair = [self.people * self.air_people_need * max(ot - self.indoortempset, 0) *
                              self.air_density * self.safecase for ot in out] #新风负荷
        self.load_window = [max(ot - self.indoortempset, 0) * self.window_k * self.windowarea for ot in out] #窗传热负荷
        self.sun_cal(mouth,day)
        #self.load_sun =

    def sun_cal(self,month,day):
        self.day_N = month * 30 + day  # 一年的第几天
        self.sin_declination = 0.39795 * math.cos(0.98563 * (self.day_N - 173))  # 太阳赤纬
        self.time_angel = numpy.linspace(-180,180,24)  # 时角
        self.h = [math.asin(math.sin(self.latitude) * self.sin_declination + math.cos(self.latitude) *
                           math.cos(math.asin(self.sin_declination)) * math.cos(t)) for t in self.time_angel] #太阳高度角
        self.A_sin =[]
        for i in range(24):
            self.A_sin.append(math.cos(math.asin(self.sin_declination)) * math.sin(self.time_angel[i]) / math.cos(self.h[i]))
        self.A_cos = [(math.sin(sh) * math.sin(self.latitude) - self.sin_declination) /
                      (math.cos(math.asin(sh)) * math.cos(self.latitude)) for sh in self.h]

r1 = Room(16,16,64,-45)
r2 = Room(8,24,64,0)
r3 = Room(16,16,64,45)
r = [r1,r2,r3]

r1.load_cal(outtemp,month,day)

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



