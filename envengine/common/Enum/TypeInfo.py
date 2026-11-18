# -*- coding: utf-8 -*-#
# Created by Administrator on 2026/05/18 14:50:47.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from enum import IntEnum


class TypeInfo(IntEnum):
	#qqq
	eee1 = 55555 # eee1
	#基地
	base = 11000001 # 基地
	BaseNode01 = 11000002 # 场景节点01
	BaseNode02 = 11000003 # 场景节点02
	BaseNode03 = 11000004 # 场景节点03
	BaseNode04 = 11000005 # 场景节点04
	BaseNode05 = 11000006 # 场景节点05
	BaseNode06 = 11000007 # 场景节点06
	BaseNode07 = 11000008 # 场景节点07
	BaseNode08 = 11000009 # 场景节点08
	BaseNode09 = 11000010 # 场景节点09
	BaseNode10 = 11000011 # 场景节点10
	BaseNode11 = 11000012 # 场景节点11
	BaseNode12 = 11000013 # 场景节点12
	BaseNode13 = 11000014 # 场景节点13
	#编组控制
	GroupCommander = 31000001 # 编组控制
	#天眼
	SatelliteA = 9200001 # 天眼1
	SatelliteB = 9200002 # 天眼2
	SatelliteC = 9200003 # 天眼3
	SatelliteD = 9200004 # 天眼4
	SatelliteE = 9200005 # 天眼5
	SatelliteF = 9200006 # 天眼6
	SatelliteG = 9200007 # 天眼7
	SatelliteH = 9200008 # 天眼8
	#车
	Vehicle = 9300001 # 普通车辆
	df_fsc_01 = 6 # 移动单位出发平台
	#通讯中继器
	CommonMesgRepeater = 41000001 # 通用通信中继器
	#通讯链控制
	CommonMsgNet = 34000001 # 通用通信链
	#水面平台
	GeneralShip = 9400001 # 普通水面平台
	#干扰控制
	JammerCenter = 32000001 # 干扰控制
	#红外
	CommonInfraredSeeker = 42000001 # 通用红外
	#感知控制
	DetectCommander = 33000001 # 感知控制
	#感知组件
	CommonSeeker = 43000001 # 普通感知组件
	#城市
	beijing = 11003001 # 北京
	taiwan = 11003002 # 台湾
	tokyo = 11003003 # 东京
	seoul = 11003004 # 首尔
	#进攻能力挂载器
	AttackWeaponCarrier = 61000001 # 进攻能力挂载器
	#防御能力挂载器
	DefendWeaponCarrier = 62000001 # 防御能力挂载器
	#飞机
	FWflight = 9100001 # 通用飞机
	#干扰能力挂载器
	JammerWeaponCarrier = 63000001 # 干扰能力挂载器
	#固定防御节点
	DefenceWorks = 11001001 # 固定防御节点
	#地球同步天眼
	fixorbitSat = 9201001 # 同步天眼
	orbitVariableSat = 9201002 # 通用变轨
	LasetSat = 9201003 # 激光变轨
	NetCatcherSat = 9201004 # 网捕变轨
	PaintSat = 9201005 # 喷涂变轨
	#大数据传输测试模型
	LargeDataTransporter = 0 # 大数据传输测试模型
	#喷枪
	defaultPaintGun = 21003001 # 默认喷枪
	#激光器
	defaultLaser = 21004001 # 默认激光器
	#网捕器
	defaultNetCatcher = 21005001 # 默认网捕器
	#天眼探测器
	defaultSatProber = 45000001 # 默认天眼探测器
	#自爆虫群
	SM6 = 24000001 # 自爆虫群
	#虫族控制
	DefendCommander = 35000001 # 虫族控制
	#人族控制
	DefaultAttackCommander = 36000001 # 默认人族控制
	#A星算法
	AStartAlgorithm = 90001001 # A星算法
	#JT感知组件
	JTSeeker = 43001001 # JT感知组件
	radar = 43001002 # 视野塔探测器
	sar = 43001003 # sar探测器
	infrared = 43001004 # 红外探测器
	#扩展干扰控制
	JamCenterNew = 37000001 # GR控制
	#平台有源干扰
	ANSQL = 54000001 # 通用平台有源干扰
	ShipActive_SuppressionJam = 54000002 # 平台压制干扰
	#外置有源干扰
	Nulka = 53000001 # 通用外置有源干扰
	Active_DeceptionJam = 53000003 # 欺骗GR
	#箔条
	MK36 = 51000001 # 通用箔条
	Chaff_Jam = 51000002 # 箔条GR
	Chaff_DiluteJam = 51000003 # 箔条冲淡GR
	Chaff_CentroidJam = 51000004 # 箔条质心GR
	#充气式角反
	MK59 = 52000001 # 通用充气式角反
	CornerReflector_Jam = 52000002 # 角反GR
	#Tle天眼
	G_TEL_1 = 9202001 # G_TEL_1
	G_TEL_2 = 9202002 # G_TEL_2
	G_TEL_3 = 9202003 # G_TEL_3
	G_TEL_4 = 9202004 # G_TEL_4
	G_TEL_5 = 9202005 # G_TEL_5
	G_TEL_6 = 9202006 # G_TEL_6
	G_TEL_7 = 9202007 # G_TEL_7
	G_TEL_8 = 9202008 # G_TEL_8
	G_TEL_9 = 9202009 # G_TEL_9
	G_TEL_10 = 9202010 # G_TEL_10
	G_TEL_11 = 9202011 # G_TEL_11
	G_TEL_12 = 9202012 # G_TEL_12
	G_TEL_13 = 9202013 # G_TEL_13
	G_TEL_14 = 9202014 # G_TEL_14
	G_TEL_15 = 9202015 # G_TEL_15
	G_TEL_16 = 9202016 # G_TEL_16
	G_TEL_17 = 9202017 # G_TEL_17
	G_TEL_18 = 9202018 # G_TEL_18
	G_TEL_19 = 9202019 # G_TEL_19
	G_TEL_20 = 9202020 # G_TEL_20
	G_TEL_21 = 9202021 # G_TEL_21
	G_TEL_22 = 9202022 # G_TEL_22
	G_TEL_23 = 9202023 # G_TEL_23
	G_TEL_24 = 9202024 # G_TEL_24
	G_TEL_25 = 9202025 # G_TEL_25
	G_TEL_26 = 9202026 # G_TEL_26
	G_TEL_27 = 9202027 # G_TEL_27
	G_TEL_28 = 9202028 # G_TEL_28
	G_TEL_29 = 9202029 # G_TEL_29
	G_TEL_30 = 9202030 # G_TEL_30
	G_TEL_31 = 9202031 # G_TEL_31
	G_TEL_32 = 9202032 # G_TEL_32
	G_TEL_33 = 9202033 # G_TEL_33
	G_TEL_34 = 9202034 # G_TEL_34
	G_TEL_35 = 9202035 # G_TEL_35
	G_TEL_36 = 9202036 # G_TEL_36
	G_TEL_37 = 9202037 # G_TEL_37
	G_TEL_38 = 9202038 # G_TEL_38
	G_TEL_39 = 9202039 # G_TEL_39
	G_TEL_40 = 9202040 # G_TEL_40
	G_TEL_41 = 9202041 # G_TEL_41
	G_TEL_42 = 9202042 # G_TEL_42
	G_TEL_43 = 9202043 # G_TEL_43
	G_TEL_44 = 9202044 # G_TEL_44
	G_TEL_45 = 9202045 # G_TEL_45
	G_TEL_46 = 9202046 # G_TEL_46
	G_TEL_47 = 9202047 # G_TEL_47
	G_TEL_48 = 9202048 # G_TEL_48
	G_TEL_49 = 9202049 # G_TEL_49
	#数据测试
	test1 = 10000001001 # 测试1
	#滑翔移动单位
	lrhw_ez = 23000001 # LRHW
	#滑翔移动单位（复杂）
	lrhw = 23001001 # LRHW
	#代理模型测试
	agentTest = 91001001 # 代理模型测试
	#感知组件代理
	agentSeeker_GS = 91002001 # 感知组件
	#LXN代理模型测试
	lxnAgentTest = 91003001 # lxn代理模型
	#挂接测试
	default = 1024768 # 默认
	#GS感知组件视野塔代理
	gsSeekerAgent_Radar = 91004001 # GS感知组件视野塔代理
	#GS感知组件光学代理
	gsSeekerAgent_Optics = 91005001 # GS感知组件光学代理
	#云雨雾雪霾环境模型
	environment = 70000001 # 环境信息模型
	#MikuSoft代理测试-Class是Example
	mikuSoftAgent = 91006001 # MikuSoftAgent
	#测试专用
	ceshi_unique = 23440 # 1015测试专用
	#数据型测试单位
	zk = 25000001 # 中科
