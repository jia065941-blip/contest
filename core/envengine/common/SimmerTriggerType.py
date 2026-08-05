# -*- coding: utf-8 -*-#
# Created by Administrator on 2026-05-18 14:50:46.
# Copyright (c) 2023 SiYuanHongRui All rights reserved.
#

from enum import IntEnum


class SimmerTriggerType(IntEnum):
    ENGINE_SPEED_CHANGE = 0  # 引擎倍速改变
    SCENE_VIEW_CHANGE = 1  # 视景视角切换
    SCENE_VIEW_FOCUS = 2  # 视景聚焦
    ATTACK = 10000  # 导弹发射
    LASER_EMIT = 10001  # 发射激光
    PAINT_EMIT = 10002  # 发射喷涂
    NET_EMIT = 10003  # 发射捕网
    DETECT_STATUS_UPDATE = 20000  # 探测器探测状态更新
    SENSOR_DETECT_UPDATE = 21000  # 探测器探测结果更新
    SENSOR_DETECT_INCREASE = 21001  # 探测器探测结果增加
    SENSOR_DETECT_DECREASE = 21002  # 探测器探测结果减少
    SENSOR_DETECT_LOCK = 21003  # 探测器锁定
    SENSOR_SCAN_UPDATE = 21004  # 导引头扫描更新
    HIT_COMPLETE = 30000  # 打击结束
    DESTROY = 40000  # 外力毁伤
    MOVEMENT_START = 50000  # 开始移动
    MOVEMENT_END = 50001  # 移动停止
    ARRIVE_TRACE_POINT = 51000  # 到达途径点
    ARRIVE_TARGET_POINT = 51001  # 到达目标点
    TRACE_POINT_UPDATE = 52000  # 修改路径点
    TARGET_POINT_UPDATE = 52001  # 修改目标点
    MISSILE_STATE_CHANGE = 60000  # 导弹飞行状态修改
    HEALTH_STATE_CHANGE = 60001  # 健康状态修改
    LIFE_POINT_CHANGE = 60002  # 生命点修改
    POWER_STATE_CHANGE = 60003  # 开关机状态修改
    RECEIVE_MESSAGE = 70000  # 接收到消息
    JAM_START = 80000  # 启动干扰
    ACTIVE_JAMMER_POWERON = 80001  # 有源干扰开机
    JAM_COMPLETE = 81000  # 干扰完成
    DEFEND_CENTER_INFO = 90000  # 拦截指控触发信息
    ATTACK_CENTER_INFO = 90001  # 进攻指控触发信息
    LAUNCH_WINDOW_OK = 90002  # 火控计算出拦截窗口
    STATISTIC_MESSAGE_UPDATE = 1000000  # 统计信息变动
    MESSAGE_CHAIN_UPDATE = 1100000  # 数据链更新
    pass
