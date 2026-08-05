# -*-coding:utf-8 -*-
class SpeedDistributor:
    """
    速度线性分配器（基于经度，但需要纬度范围来定义区域）
    """

    def __init__(self,
                 min_lon: float, max_lon: float,  # 经度范围（决定速度）
                 min_lat: float, max_lat: float,  # 纬度范围（只定义区域，不影响速度）
                 min_speed: float = 800, max_speed: float = 1200):
        """
        参数:
            min_lon: 最小经度（最左边）
            max_lon: 最大经度（最右边）
            min_lat: 最小纬度（最下边，仅用于区域判断）
            max_lat: 最大纬度（最上边，仅用于区域判断）
            min_speed: 最左边的速度
            max_speed: 最右边的速度
        """
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.lon_range = max_lon - min_lon
        self.speed_range = max_speed - min_speed

    def get_speed(self, lon: float) -> float:
        """根据经度获取速度（纬度不影响速度）"""
        if self.lon_range == 0:
            return (self.min_speed + self.max_speed) / 2

        t = (lon - self.min_lon) / self.lon_range
        t = max(0, min(1, t))
        return self.min_speed + t * self.speed_range

    def is_in_region(self, lon: float, lat: float) -> bool:
        """检查点是否在区域内"""
        return (self.min_lon <= lon <= self.max_lon and
                self.min_lat <= lat <= self.max_lat)


# ========== 使用 ==========

# # 创建分配器（定义二维区域）
# distributor = SpeedDistributor(
#     min_lon=-100, max_lon=120,  # 经度范围 100°E ~ 120°E
#     min_lat=30, max_lat=40,  # 纬度范围 30°N ~ 40°N
#     min_speed=3000, max_speed=5000
# )
#
# # 获取速度（只需要传经度）
# speed1 = distributor.get_speed(-100)  # 最左边 → 3000
# speed2 = distributor.get_speed(-110)  # 中间 → 4000
# speed3 = distributor.get_speed(120)  # 最右边 → 5000
#
# print(f"经度100°: {speed1:.0f}")
# print(f"经度110°: {speed2:.0f}")
# print(f"经度120°: {speed3:.0f}")
#
# # 检查点是否在区域内
# in_region = distributor.is_in_region(105, 35)  # True