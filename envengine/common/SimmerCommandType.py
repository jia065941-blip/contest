# -*- coding: utf-8 -*-#
# Created by Administrator on 2026/05/18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from enum import IntEnum


class SimmerCommandType(IntEnum):
    """
    Attributes:
        ATTACK_COMMANDER_START: 人族控制初始化
    """
    MISSILE_LAUNCH = 200  # 人族单位出发
    INTERCEPTOR_LAUNCH = 201  # 自爆虫群释放
    MISSILE_GUIDANCE = 202  # 人族单位目标引导
    INTERCEPTOR_GUIDANCE = 203  # 自爆虫群目标引导
    MISSILE_CHANGE_ROUTE = 204  # 人族单位修改航迹
    LASER_EMIT = 205  # 释放激光
    PAINT_EMIT = 206  # 执行喷涂
    NET_EMIT = 207  # 投放捕网
    MOVE = 300  # 开始移动
    PATROL_ALONG_ROUTE = 301  # 按路径点巡逻
    PATROL_CIRCLE = 302  # 大圆巡逻
    CHANGE_ORBIT = 303  # 天眼变轨
    GROUP_MOVE = 310  # 编组移动
    GROUP_PATROL_ROUTE = 311  # 编组按路径巡逻
    GROUP_PATROL_CIRCLE = 312  # 编组大圆巡逻
    GROUP = 400  # 编组
    CHANGE_GROUP = 401  # 修改编组
    DISSOLUTION = 402  # 解散编组
    EXEC_GROUP = 410  # 单机执行编组指令
    START_DETECT = 500  # 启动探测
    DAMAGE = 600  # 伤害
    DESTROY = 700  # 摧毁
    EASY_EVENT_SEND = 800  # 简易事件转发
    MESSAGE_CHAIN_CREATE = 802  # 通讯链创建
    MESSAGE_CHAIN_DESTROY = 803  # 删除通讯链路
    MESSAGE_CHAIN_ACTIVE = 804  # 通讯链路激活
    HEALTH_STATE_CHANGE = 900  # 健康状态切换
    POWER_STATE_CHANGE = 901  # 电源状态切换
    b = 999  # b
    INTERCEPTOR_INFO = 1000  # 虫族控制指令
    CE_SHI = 1005  # 测试指令勿删
    RELEASE_JAM = 1100  # 释放干扰
    GR_RELEASE = 1101  # GR释放指令
    CHANGE_JAMPOS = 1103  # 干扰位置改变
    RETRIEVE_CHILDREN = 1200  # 回收子物体
    RELEASE_CHILDREN = 1201  # 释放子物体
    SET_RELATED_ENTITY = 1202  # 设置相关实体
    DETECT_ENTITY = 1300  # 感知控制信息
    ENGINE_SPEED_SET = 1400  # 引擎倍速切换
    SCENE_VIEW_CHANGE = 1401  # 视景视角切换
    SCENE_VIEW_FOCUS = 1402  # 视景聚焦
    CHILD_SYNC = 1403  # 子物体位置速度同步
    JAMMMER_COMMANDER_START = 2100  # 干扰控制开始工作
    GR_COMMAND_START = 2101  # 干扰控制开始工作指令
    DEFEND_COMMANDER_START = 2200  # 虫族控制初始化
    TRANS_LAUNCH_WINDOW = 2202  # 传递响应窗口
    ATTACK_COMMANDER_START = 2300  # 人族控制初始化
    EnvironmentInfoSet = 3000  # 环境信息设置
    EntityInfoSet = 3001  # 实体信息设置
    TransmissionInfoSet = 3002  # 传输信息设置
    InterfereInfoSet = 3003  # 干扰信息设置
    TargetCharacterInfo = 3004  # 目标属性设置
    ChangePose = 3005  # 改变实体位姿
    DETECT_STATUS_UPDATE = 3006  # 探测器探测状态更新
    SET_DESIRED_ACC_Z = 3007  # 设置飞行器侧向加速度
    DEPLOY = 3008  # 部署
    DEPLOY_COMPLETED = 3009  # 部署完成
    RADAR_DETECT_STATUS_UPDATE = 3010  # 视野塔探测状态更新
    EXECUTE_DETECTION = 3011  # 执行探测
    SET_DESIRED_VEL_X = 3012  # 设置飞行器轴向速度
    EXECUTE_SATELLITE_DETECTION = 3013  # 请求天眼探测
    CHANGE_MISSILE_TARGET = 3014  # 修改移动单位目标点


