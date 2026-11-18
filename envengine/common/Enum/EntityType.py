# -*- coding: utf-8 -*-#
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from enum import IntEnum


class EntityType(IntEnum):
	pppp = 855 # pppp
	MotionCarrier = 9000 # 运动载体
	FixedWingFlight = 9100 # 飞机
	Satellite = 9200 # 天眼
	FixedSatellite = 9201 # 地球同步天眼
	TleSatellite = 9202 # Tle天眼
	MotionPlatform = 9300 # 车
	GeneralShip = 9400 # 水面平台
	oooo = 9888 # ooo
	Entity = 10000 # 实体
	Ceshi = 10005 # 测试专用
	Base = 11000 # 基地
	DefenceWorks = 11001 # 固定防御节点
	CommandCenterSkillModel = 11002 # 控制中心技能模型
	City = 11003 # 城市
	BH_Missile = 12345 # BH移动单位
	Weapon = 20000 # 能力单位
	CruiseMissile = 21000 # 毁灭者
	CruiseMissile2 = 21001 # 突击者
	CruiseMissileWRJ = 21002 # 清道夫
	Painter = 21003 # 喷枪
	Laser = 21004 # 激光器
	NetCather = 21005 # 网捕器
	RocketMissile = 21006 # 加速移动单位
	BallisticMissile = 22000 # 抛物线移动单位
	GlidingMissile = 23000 # 滑翔移动单位
	GlidingMissile2 = 23001 # 滑翔移动单位（复杂）
	BH_missile_1 = 23456 # BH移动单位
	Interceptor = 24000 # 自爆虫群
	DataMissile = 25000 # 数据型测试单位
	Commander = 30000 # 控制
	GroupCommander = 31000 # 编组控制
	JammerCommander = 32000 # 干扰控制
	DetectCommander = 33000 # 感知控制
	MsgNet = 34000 # 通讯链控制
	Controller = 34567 # 控制
	DefendCommander = 35000 # 虫族控制
	AttackCommander = 36000 # 人族控制
	JammerCenterCommander = 37000 # 扩展干扰控制
	Sensor = 40000 # 传感器
	MsgRepeater = 41000 # 通讯中继器
	InfaredSeeker = 42000 # 红外
	Seeker = 43000 # 感知组件
	JTC_Sensor_Seeker = 43001 # JT感知组件
	NormalRadar = 44000 # 视野塔
	SatelliteProb = 45000 # 天眼探测器
	Controller_1 = 45678 # 控制_1
	GeneralJammerS = 50000 # 干扰
	ChaffS = 51000 # 箔条
	AngularReversalS = 52000 # 充气式角反
	NarkaS = 53000 # 外置有源干扰
	ShipboardActive = 54000 # 平台有源干扰
	qqq = 55555 # qqq
	WeaponCarrier = 60000 # 能力挂载器
	AttackWeaponCarrier = 61000 # 进攻能力挂载器
	DefendWeaponCarrier = 62000 # 防御能力挂载器
	JammerWeaponCarrier = 63000 # 干扰能力挂载器
	EnvironmentModel = 70000 # 云雨雾雪霾环境模型
	AlgorithmAgent = 90000 # 算法代理
	AStartAlgorithmAgent = 90001 # A星算法
	SimulatorAgent = 91000 # 模型代理
	AgentTest = 91001 # 代理模型测试
	SeekerAgent_GS = 91002 # 感知组件代理
	LXN_AgentTest = 91003 # LXN代理模型测试
	GSSeekerAgent_Radar = 91004 # GS感知组件视野塔代理
	GSSeekerAgent_Optics = 91005 # GS感知组件光学代理
	MikuSoftAgent = 91006 # MikuSoft代理测试-Class是Example
	LargeDataTransporter = 1000000 # 大数据传输测试模型
	TestData = 1000001 # 数据测试
	LinkTestSimulator = 19201080 # 挂接测试
	LL = 33333333 # 兼容测试实体
