# -*-coding:utf-8 -*-
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from envengine.sdk.base_struct.profile import Profile
from envengine.sdk.base_struct.profile.Environment import Environment
from envengine.sdk.base_struct.profile.Imagine import Imagine
from envengine.sdk.base_struct.Entity.EntityExt import EntityExt
from envengine.sdk.base_struct.Entity.Entity import Entity
from envengine.sdk.base_struct.profile.SimulatorConfig import SimulatorConfig
from envengine.sdk.base_struct.Message.Area import Area
from envengine.sdk.base_struct.Message.MapArea import MapArea


class ScenarioGenerator:
    """想定生成器"""

    # 模型类型映射
    MODEL_TYPES = {
        21000: {"name": "高性能飞行器", "type_id": 21000001},
        21001: {"name": "巡航弹v2", "type_id": 21001001},
        21002: {"name": "低性能巡航弹", "type_id": 21002001},
        9202: {"name": "卫星", "type_id": 9202001},
        9400: {"name": "目标", "type_id": 9400001},
        9500: {"name": "无人船", "type_id": 9500001},
        9600: {"name": "拦截阵地", "type_id": 9600001},
        24000: {"name": "拦截弹", "type_id": 24000001},
        44000: {"name": "普通雷达", "type_id": 44000001},
        35000: {"name": "拦截指控", "type_id": 35000001},
    }

    # 动态库路径映射
    DYNAMIC_LIBRARY_MAP = {
        24000: "./Packages/Interceptor/libInterceptorS.so",
        35000: "./Packages/DefendCommander/libDefendCommanderS.so",
        21000: "./Packages/CruiseMissile/libCruiseMissileHS.so",
        9202: "./Packages/TleSatellite/libTleSatelliteS.so",
        21001: "./Packages/CruiseMissile2/libCruiseMissileMS.so",
        9400: "./Packages/GeneralShip/libGeneralShipS.so",
        21002: "./Packages/CruiseMissileL/libCruiseMissileLS.so",
        44000: "./Packages/NormalRadar/libNormalRadarS.so",
        9500: "./Packages/UnmannedShip/libUnmannedShipS.so",
        9600: "./Packages/InterceptField/libInterceptFieldS.so",
    }

    def __init__(self):
        self.profile = Profile()
        self._init_environment()
        self.entity_id_counter = 2

    def _init_environment(self):
        """初始化环境配置"""
        env = self.profile.environmentProfile
        env.mqttIp = "127.0.0.1"
        env.domain = 232
        env.subDomain = "50"
        env.projectName = "XXX"
        env.engineName = "platform"
        env.engineId = 1
        env.role = 0
        env.parentId = -1
        env.parentName = ""
        env.childrenId = []
        env.childrenName = []
        env.publishData = []
        env.subscribeData = []
        env.patchDynamicLibrary = "./PatchShared.dll"
        env.withPython = 0

        # 配置动态库
        env.dynamicLibraryConfigs = []
        for entity_type, lib_path in self.DYNAMIC_LIBRARY_MAP.items():
            if entity_type in self.MODEL_TYPES:
                config = SimulatorConfig(
                    entityType=entity_type,
                    entityTypeId=[self.MODEL_TYPES[entity_type]["type_id"]],
                    dynamicLibraryPath=lib_path
                )
                env.dynamicLibraryConfigs.append(config)

    def create_entity(self, entity_type: int, x: float = 0, y: float = 0,
                      name_prefix: str = "", side_id: int = 0, parent_id: int = -1,
                      survive_points: int = 1500, threat_level: int = 0,
                      is_visible: bool = True, custom_name: str = None,
                      custom_en_name: str = None) -> EntityExt:
        """创建单个实体"""
        type_id = self.MODEL_TYPES[entity_type]["type_id"]
        entity_id = self.entity_id_counter
        self.entity_id_counter += 1

        model_info = self.MODEL_TYPES.get(entity_type, {"name": "未知"})

        # 如果有自定义名称则使用，否则自动生成
        if custom_name:
            name_chn = custom_name
            name_en = custom_en_name if custom_en_name else custom_name.replace(' ', '_')
        else:
            name_chn = f"{model_info['name']}_{type_id}_{entity_id:03d}"
            name_en = f"{model_info['name'].replace('飞行器', 'Cruise').replace('巡航弹', 'CruiseMissile').replace('无人机', 'UAV').replace('拦截弹', 'Interceptor').replace('拦截指控', 'DefendCommander')}_{entity_id:03d}"

        if name_prefix and not custom_name:
            name_chn = f"{name_prefix}_{name_chn}"

        entity = Entity(
            id=entity_id,
            entityType=entity_type,
            typeId=type_id,
            nameChn=name_chn,
            nameEn=name_en,
            lla={"x": x, "y": y, "z": 0},
            posEcf={"x": 0, "y": 0, "z": 0},
            velEcf={"x": 0, "y": 0, "z": 0},
            velNue={"x": 0, "y": 0, "z": 0},
            parentId=parent_id,
            childrenId=[],
            sideId=side_id,
            isVisible=is_visible,
            survivePoints=survive_points,
            threatLevel=threat_level,
            step=1000,
            time=1783391450000
        )

        return EntityExt(entity=entity, external="")

    # ==================== 分布函数 ====================
    # 用户可以在这里修改分布逻辑

    def get_position(self, area_config: Dict) -> Tuple[float, float]:
        """
        根据区域配置生成位置
        用户可以直接修改此函数来改变分布方式

        参数:
            area_config: 区域配置字典，包含:
                - x_min: X轴最小值
                - x_max: X轴最大值
                - y_min: Y轴最小值
                - y_max: Y轴最大值
                - 用户可自行添加其他参数

        返回:
            (x, y)
        """
        x_min = area_config.get("x_min", -4.0)
        x_max = area_config.get("x_max", -0.5)
        y_min = area_config.get("y_min", -1.8)
        y_max = area_config.get("y_max", 1.8)

        # ===== 用户可在此修改分布逻辑 =====
        # 当前使用均匀分布
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        return x, y

    def generate_red_entities(self, red_config: List[Dict]) -> List[EntityExt]:
        """
        生成红方实体 - 左半边
        同时根据配置设置 redArea 和 mapArea
        """
        red_entities = []

        # 初始化区域配置
        area_21000_21001 = None  # 用于 redArea.coordinatesHM
        area_21002 = None  # 用于 redArea.coordinates

        for config in red_config:
            entity_type = config["type"]
            count = config.get("count", 1)
            area_config = config.get("area", {})
            survive_points = config.get("survive_points", 1500)
            threat_level = config.get("threat_level", 0)
            custom_name = config.get("custom_name", None)
            custom_en_name = config.get("custom_en_name", None)
            is_visible = config.get("is_visible", True)

            # 保存区域配置用于后续设置 redArea
            if entity_type in [21000, 21001]:
                area_21000_21001 = area_config
            elif entity_type == 21002:
                area_21002 = area_config

            # 生成实体
            for i in range(count):
                x, y = self.get_position(area_config)
                entity = self.create_entity(
                    entity_type, x, y, "红方",
                    side_id=0,
                    survive_points=survive_points,
                    threat_level=threat_level,
                    is_visible=is_visible,
                    custom_name=custom_name,
                    custom_en_name=custom_en_name
                )
                red_entities.append(entity)
                self.profile.imagineProfile.entityList.append(entity)

        # 设置 redArea
        imagine = self.profile.imagineProfile

        # 创建 redArea 对象
        red_area = Area(type="Polygon")

        # 设置 redArea.coordinatesHM (21000和21001的区域 - 左半边)
        if area_21000_21001:
            x_min = area_21000_21001.get("x_min", -4.0)
            x_max = area_21000_21001.get("x_max", -2.0)
            y_min = area_21000_21001.get("y_min", -1.8)
            y_max = area_21000_21001.get("y_max", 1.8)

            polygon_hm = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]
            red_area.coordinatesHM = [polygon_hm]

        # 设置 redArea.coordinates (21002的区域 - 左半边偏右)
        if area_21002:
            x_min = area_21002.get("x_min", -2.0)
            x_max = area_21002.get("x_max", -0.5)
            y_min = area_21002.get("y_min", -1.8)
            y_max = area_21002.get("y_max", 1.8)

            polygon = [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ]
            red_area.coordinates = [polygon]

        # 如果没有任何区域被设置，使用默认值
        if not red_area.coordinatesHM and not red_area.coordinates:
            # 默认区域 - 左半边
            default_polygon_hm = [
                [-4.0, -1.8],
                [-2.0, -1.8],
                [-2.0, 1.8],
                [-4.0, 1.8]
            ]
            red_area.coordinatesHM = [default_polygon_hm]

            default_polygon = [
                [-2.0, -1.8],
                [-0.5, -1.8],
                [-0.5, 1.8],
                [-2.0, 1.8]
            ]
            red_area.coordinates = [default_polygon]

        imagine.redArea = red_area

        # 设置 mapArea - 整体范围: X轴-4到4, Y轴-2到2
        imagine.mapArea = MapArea(
            lonMin=-4.0,
            lonMax=4.0,
            latMin=-2.0,
            latMax=2.0
        )

        return red_entities

    def generate_blue_entities(self, blue_config: List[Dict]) -> List[EntityExt]:
        """
        生成蓝方实体 - 右半边

        config 字段:
            - type:       实体类型
            - x, y:       实体的具体坐标 (必须提供, 不再支持范围随机)
            - interceptors: 挂载拦截弹数量
            - radar_type:   挂载雷达类型
            - survive_points / threat_level / is_visible / custom_name / custom_en_name
        说明:
            雷达、拦截弹等子节点坐标与父节点完全一致 (不再加随机偏移)。
        """
        blue_entities = []
        interceptor_counter = 1  # 拦截弹全局计数器

        for config in blue_config:
            entity_type = config["type"]

            # 使用具体坐标 (蓝方配置必须提供具体的 x/y)
            x = config["x"]
            y = config["y"]


            interceptors = config.get("interceptors")
            radar_type = config.get("radar_type")
            survive_points = config.get("survive_points", 1500)
            threat_level = config.get("threat_level", 0)
            custom_name = config.get("custom_name", None)
            custom_en_name = config.get("custom_en_name", None)
            is_visible = config.get("is_visible", True)

            # 创建主实体
            entity = self.create_entity(
                entity_type, x, y, "蓝方",
                side_id=1,
                survive_points=survive_points,
                threat_level=threat_level,
                is_visible=is_visible,
                custom_name=custom_name,
                custom_en_name=custom_en_name
            )
            blue_entities.append(entity)
            self.profile.imagineProfile.entityList.append(entity)

            # 挂载拦截弹 (24000) - 命名格式: 拦截弹-id-父节点id, 坐标与父节点一致
            if interceptors:
                for i in range(interceptors):
                    interceptor_name = f"拦截弹-{interceptor_counter}-{entity.entity.id}"
                    interceptor_counter += 1

                    interceptor = self.create_entity(
                        24000,
                        x, y,
                        name_prefix="",
                        side_id=1,
                        parent_id=entity.entity.id,
                        survive_points=1500,
                        is_visible=False,
                        custom_name=interceptor_name,
                        custom_en_name=f"Interceptor_{interceptor_counter - 1}_{entity.entity.id}"
                    )
                    entity.entity.childrenId.append(interceptor.entity.id)
                    self.profile.imagineProfile.entityList.append(interceptor)

            # 挂载雷达 (44000) - 坐标与父节点一致
            if radar_type and radar_type in self.MODEL_TYPES:
                radar = self.create_entity(
                    radar_type,
                    x, y,
                    "蓝方_雷达",
                    side_id=1,
                    parent_id=entity.entity.id,
                    survive_points=1500,
                    is_visible=False,
                    custom_name=config.get("radar_name", f"普通雷达_{radar_type}_{entity.entity.id}"),
                    custom_en_name=config.get("radar_en_name", f"NormalRadar_{radar_type}_{entity.entity.id}")
                )
                radar.external = '{"raderExternal":{"range":{"x":0,"y":0,"z":0},"distance":350000,"scanSpeed":5}}'
                entity.entity.childrenId.append(radar.entity.id)
                self.profile.imagineProfile.entityList.append(radar)

        return blue_entities

    def generate_defend_commander(self) -> EntityExt:
        """
        生成拦截指控实体 (35000) - 位于中间
        """
        defend_commander = self.create_entity(
            35000,
            0.0,
            0.0,
            name_prefix="",
            side_id=0,
            survive_points=1500,
            threat_level=0,
            custom_name="拦截指控_35000001001",
            custom_en_name="DefendCommander_35000001001"
        )
        return defend_commander

    def generate_scenario(self, red_config: List[Dict], blue_config: List[Dict]):
        """
        生成想定
        """
        # 生成红方实体
        print("正在生成红方实体...")
        red_entities = self.generate_red_entities(red_config)

        # 生成蓝方实体
        print("正在生成蓝方实体...")
        blue_entities = self.generate_blue_entities(blue_config)

        # 生成拦截指控实体 (35000)
        print("正在生成拦截指控实体...")
        defend_commander = self.generate_defend_commander()
        self.profile.imagineProfile.entityList.append(defend_commander)

        # 设置想定参数
        imagine = self.profile.imagineProfile
        imagine.simStep = 1000
        imagine.simTime = 1783391450000
        imagine.simEndLogicTime = 50000000
        imagine.logicTime = 0
        imagine.rules = []
        imagine.eventList = []
        imagine.taskList = []

        print(f"\n想定生成完成!")
        print(f"红方实体: {len(red_entities)} 个")
        print(f"蓝方实体: {len(blue_entities)} 个")
        print(f"拦截指控: 1 个")
        print(f"总实体数: {len(self.profile.imagineProfile.entityList)} 个")

        return red_entities, blue_entities, [defend_commander]

    def save_to_file(self, filename: str = "scenario.json"):
        """保存想定到文件"""
        data = self.profile.to_dict()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"想定已保存到: {filename}")


# ==================== 用户配置区域 ====================
# mapArea 整体范围: 经度[-4, 4], 纬度[-2, 2]
# 红方在左半边 (经度 -4 到 -0.5)
# 蓝方在右半边 (经度 0.5 到 4)

LAND_AREA = {
    "x_min": -3.8,
    "x_max": -2.2,
    "y_min": -1.5,
    "y_max": 1.5
}

RED_CONFIG = [
    # 高性能飞行器 (21000) - 左侧区域 (将保存到 redArea.coordinatesHM)
    {
        "type": 21000,
        "count": 50,
        "survive_points": 1,
        "threat_level": 0,
        "area": LAND_AREA
    },
    # 低性能飞行器 (21001) - 左侧区域 (将保存到 redArea.coordinatesHM)
    {
        "type": 21001,
        "count": 100,
        "survive_points": 1500,
        "threat_level": 0,
        "area": LAND_AREA
    },
    # 无人机 (21002) - 左半边偏右区域 (将保存到 redArea.coordinates)
    {
        "type": 21002,
        "count": 200,
        "survive_points": 1,
        "threat_level": 5,
        "area": {
            "x_min": -2.0,
            "x_max": -0.8,
            "y_min": -1.5,
            "y_max": 1.5
        }
    },
    # 卫星 (9202)
    {
        "type": 9202,
        "count": 1,
        "survive_points": 1500,
        "threat_level": 0,
        "area": {
            "x_min": -0.6,
            "x_max": -0.6,
            "y_min": 0.8,
            "y_max": 0.8
        },
        "is_visible": True
    },
]

# 蓝方配置 - 在右半边 (经度 0.5 到 3.8)
BLUE_CONFIG = [
    # 拦截阵地1 - 挂载50枚拦截弹和雷达
    {
        "type": 9600,
        "x": 1.8,
        "y": 0.9,
        "survive_points": 30,
        "threat_level": 0,
        "interceptors": 50,
        "radar_type": 44000,
    },
    # 拦截阵地2 - 挂载50枚拦截弹和雷达
    {
        "type": 9600,
        "x": 2.2,
        "y": -0.8,
        "survive_points": 30,
        "threat_level": 0,
        "interceptors": 50,
        "radar_type": 44000,
    },
    # 目标1
    {
        "type": 9400,
        "x": 1.9,
        "y": 0.7,
        "survive_points": 60,
        "threat_level": 0,
        "interceptors": None,
        "radar_type": None,
    },
    # 目标2
    {
        "type": 9400,
        "x": 2.6,
        "y": -0.2,
        "survive_points": 60,
        "threat_level": 0,
        "interceptors": None,
        "radar_type": None,
    },
    # 目标3
    {
        "type": 9400,
        "x": 2.3,
        "y": -1.3,
        "survive_points": 60,
        "threat_level": 0,
        "interceptors": None,
        "radar_type": None,
    },
    # 无人船1 - 挂载50枚拦截弹
    {
        "type": 9500,
        "x": 1.7,
        "y": 1.7,
        "survive_points": 10,
        "threat_level": 0,
        "interceptors": 50,
        "radar_type": None,
    },
    # 无人船2 - 挂载50枚拦截弹
    {
        "type": 9500,
        "x": 3.1,
        "y": 0.2,
        "survive_points": 10,
        "threat_level": 0,
        "interceptors": 50,
        "radar_type": None,
    },
]


def generate_blue_config() -> List[Dict]:
    """
    生成标准蓝方配置 (独立函数, 与 ScenarioGenerator 解耦)

    返回一份 blue_config 列表, 每一项对应一个实体。每项都带有具体的坐标 x/y
    (而非范围), generate_blue_entities() 会直接使用该坐标, 且挂载的雷达/拦截弹
    坐标与父节点完全一致。

    部署计划 (右半边 经度 0.5~4.0, 纬度 -2.0~2.0):
        - 上区域 (拦截阵地 9600): 20 个, 各带 1 部雷达 (44000) 50 个拦截弹
        - 下区域 (拦截阵地 9600): 20 个, 各带 1 部雷达 (44000) 50 个拦截弹
        - 左边区域 (无人船 9500): 50 个, 各带 1 部雷达 (44000) 50 个拦截弹
        - 中间区域 (目标 9400):   10 个, 无雷达

    区域划分:
        - 上区域:   纬度  0.2~1.8, 经度 0.8~3.8
        - 下区域:   纬度 -1.8~-0.2, 经度 0.8~3.8
        - 左边区域: 经度  0.5~1.8 (靠近中线), 纬度 -1.8~1.8
        - 中间区域: 经度  1.8~3.0, 纬度 -0.8~0.8
    """
    blue_config = []

    # 上区域拦截阵地 20 个 (每个各带雷达, 拦截弹共 25 枚: 其中 5 个带 2 枚, 15 个带 1 枚)
    for i in range(20):
        blue_config.append({
            "type": 9600,
            "x": random.uniform(0.8, 3.8),
            "y": random.uniform(0.2, 1.8),
            "survive_points": 30,
            "threat_level": 0,
            "interceptors": 50,
            "radar_type": 44000,
        })

    # 下区域拦截阵地 20 个 (同上, 拦截弹共 25 枚)
    for i in range(20):
        blue_config.append({
            "type": 9600,
            "x": random.uniform(0.8, 3.8),
            "y": random.uniform(-1.8, -0.2),
            "survive_points": 30,
            "threat_level": 0,
            "interceptors": 50,
            "radar_type": 44000,
        })

    # 左边区域无人船 50 个 (每个各带雷达, 带拦截弹)
    for _ in range(50):
        blue_config.append({
            "type": 9500,
            "x": random.uniform(0.5, 1.8),
            "y": random.uniform(-1.8, 1.8),
            "survive_points": 10,
            "threat_level": 0,
            "interceptors": 50,
            "radar_type": 44000,
        })

    # 中间目标 10 个 (无雷达、无拦截弹)
    for _ in range(10):
        blue_config.append({
            "type": 9400,
            "x": random.uniform(1.8, 3.0),
            "y": random.uniform(-0.8, 0.8),
            "survive_points": 60,
            "threat_level": 0,
            "interceptors": None,
            "radar_type": None,
        })

    return blue_config


# 输出文件名
OUTPUT_FILE = "scenario.json"


# ==================== 主程序 ====================

def main():
    generator = ScenarioGenerator()

    # ===== 蓝方配置开关 =====
    # 取消注释下一行 -> 启用自动生成的标准蓝方配置 (覆盖下方原 BLUE_CONFIG)
    # 注释掉下一行     -> 保留原 BLUE_CONFIG 手动配置
    # BLUE_CONFIG = generate_blue_config()

    # 打印配置信息
    print("=" * 60)
    print("想定生成器")
    print("=" * 60)
    print("\nmapArea 整体范围: 经度[-4, 4], 纬度[-2, 2]")
    print("红方区域: 左半边 (经度 -4 到 -0.5)")
    print("蓝方区域: 右半边 (经度 0.5 到 4)")
    print("拦截指控: 位于 (0, 0)")
    print("=" * 60)

    # 统计红方实体数量
    red_total = sum(config["count"] for config in RED_CONFIG)
    blue_total = len(BLUE_CONFIG)
    interceptor_total = sum(config.get("interceptors", 0) or 0 for config in BLUE_CONFIG)

    print(f"\n红方实体总数: {red_total}")
    for config in RED_CONFIG:
        model_name = generator.MODEL_TYPES[config["type"]]["name"]
        area = config.get("area", {})
        print(f"  - {model_name}: {config['count']} 个 (经度[{area.get('x_min', 0):.1f}, {area.get('x_max', 0):.1f}])")

    print(f"\n蓝方实体总数: {blue_total}")
    for config in BLUE_CONFIG:
        model_name = generator.MODEL_TYPES[config["type"]]["name"]
        interceptors = config.get("interceptors", 0) or 0
        radar = "有雷达" if config.get("radar_type") else "无雷达"
        print(
            f"  - {model_name}: 经度[{config['x']:.1f}, {config['x']:.1f}], 拦截弹 {interceptors} 枚, {radar}")

    print(f"\n拦截弹总数: {interceptor_total}")
    print(f"拦截指控: 1 个")
    print("=" * 60)

    # 生成想定
    generator.generate_scenario(
        red_config=RED_CONFIG,
        blue_config=BLUE_CONFIG
    )

    # 保存到文件
    generator.save_to_file(OUTPUT_FILE)

if __name__ == "__main__":
    
    main()