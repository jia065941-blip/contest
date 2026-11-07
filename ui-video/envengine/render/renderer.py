# -*-coding:utf-8 -*-
from io import BytesIO
from collections import OrderedDict
import math
import os
from pathlib import Path
import threading
import time

# 控制台留给仿真运行提示，渲染库导入时默认隐藏版本与社区问候。
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
from pygame.draw import rect
from pyproj import Transformer
import typing


class Renderer:
    """
    实时渲染器 - 独立线程运行pygame窗口
    """

    BACKGROUND_COLOR = (232, 240, 246)
    GRID_COLOR = (207, 220, 229)
    TEXT_COLOR = (38, 50, 65)
    MUTED_TEXT_COLOR = (85, 100, 118)
    PANEL_COLOR = (255, 255, 255)
    PANEL_BORDER_COLOR = (199, 208, 220)
    SIDE_COLORS = {0: (196, 52, 75), 1: (45, 109, 190)}
    NEUTRAL_COLOR = (97, 110, 128)
    # 保留单位类型的色相，用深色线条提高浅底和阵营底圆上的对比度。
    TYPE_COLORS = {
        21000: (177, 34, 66),
        21001: (0, 103, 145),
        21002: (246, 190, 32),
        9202: (112, 58, 180),
        9400: (180, 107, 16),
        9500: (100, 56, 169),
        9600: (20, 112, 66),
        24000: (244, 48, 92),
    }
    MARKER_BACKGROUND_RADIUS = 28
    EYE_MARKER_SIZE = 75
    EYE_BACKGROUND_RADIUS = 38
    EYE_EDGE_COLOR = (136, 24, 45)
    SWARM_MARKER_SIZE = 34
    SWARM_BACKGROUND_RADIUS = 20

    # 实时高密度态势使用紧凑模型；组织方静态预览仍可沿用较大图标。
    LIVE_MARKER_SIZES = {
        21000: 26,  # H: compact missile, slightly smaller than L
        21001: 24,  # M: compact missile, slightly smaller than L
        21002: 30,  # L: enlarged UAV for dense-scene visibility
        9400: 62,   # mother hive: largest hive
        9500: 28,   # hidden hive: smallest hive
        9600: 38,   # sentry hive: middle size
        24000: 19,
    }
    LIVE_BACKGROUND_RADII = {
        21000: 18,
        21001: 12,
        21002: 9,
        9400: 19,
        9500: 8,
        9600: 11,
        24000: 7,
    }
    TYPE_LABELS = {
        21000: "Destroyer (H)",
        21001: "Assault (M)",
        21002: "Scavenger (L)",
        9202: "Satellite",
        9400: "Hive",
        9500: "Hidden hive",
        9600: "Sentry hive",
        24000: "Interceptor",
    }
    STAGE_LABELS = {
        1: "Launch",
        2: "Rise",
        3: "Cruise",
        4: "Terminal",
        5: "Explode",
    }
    HIVE_SPRITE_FILES = {
        9400: "mother_hive.png",
        9500: "hidden_hive.png",
        9600: "sentry_hive.png",
    }
    HIVE_TYPE_COLORS = {
        9400: (230, 148, 42),   # mother hive: amber
        9500: (139, 92, 246),  # hidden hive: violet
        9600: (34, 197, 94),   # sentry hive: green
    }
    EXPLOSION_DURATION_SECONDS = 1.1

    def __init__(self, lon_min, lon_max, lat_min, lat_max, fps=60,
                 sim_step_seconds=1.0, motion_fps=24):
        self.entities_data = {}
        self.red_area = []
        self.running = False
        self.thread = None
        self._ready_event = threading.Event()
        self._frame_condition = threading.Condition()
        self._submitted_frame = 0
        self._rendered_frame = 0
        self._pending_step = 0
        self.current_step = 0
        self._last_submit_time = None
        self._step_interval_ms = 0.0
        self._render_time_ms = 0.0
        self._present_rate_hz = 0.0
        self._last_present_time = None
        self._display_phase = 0.0
        self._last_motion_present = 0.0
        self.sim_step_seconds = max(1e-6, float(sim_step_seconds))
        self.motion_fps = max(1, int(motion_fps))
        self.background_image_path = (
            Path(__file__).resolve().parents[2]
            / "assets/backgrounds/background.png"
        )
        self.ui_font_path = (
            Path(__file__).resolve().parents[2]
            / "assets/fonts/NotoSansCJK-Regular.ttc"
        )
        self.hive_sprite_dir = (
            Path(__file__).resolve().parents[2] / "assets/hives"
        )
        self._background_master = None
        self.background_opacity = 1.0
        self.geographic_layout = None
        # Purely illustrative detection ring; it is not a physical distance scale.
        self.range_ring_radius_px = 56
        # 当前底图已按最终1920x1080视口排版，保持1:1像素绘制。
        self.background_rotation_deg = 0.0
        self.background_offset_y = 0.0
        # 二维战术视图可独立旋转；正角度表示屏幕视觉上的逆时针旋转。
        self.view_rotation_deg = 0.0
        # 地图配置：横纵坐标保持固定比例，避免渲染变形
        self.MIN_WINDOW_SIZE = 500
        self.LON_MIN, self.LON_MAX = lon_min, lon_max
        self.LAT_MIN, self.LAT_MAX = lat_min, lat_max
        # 3D地面使用仿真场景的固定边界；2D视图仍可随窗口尺寸扩展取景范围。
        self.SCENE_LON_MIN, self.SCENE_LON_MAX = lon_min, lon_max
        self.SCENE_LAT_MIN, self.SCENE_LAT_MAX = lat_min, lat_max
        self.lon_per_pixel = self.lat_per_pixel = 0 # 每像素对应的横纵坐标跨度（初始化后保持不变）
        self.MAP_W, self.MAP_H = max(1000, self.MIN_WINDOW_SIZE), max(800, self.MIN_WINDOW_SIZE)
        # 初始化坐标比例并记录地图中心
        self._init_lonlat_pixel_ratio()
        self._recalculate_lonlat_boundary()

        # ========== 缩放和平移参数 ==========
        self.zoom = 1.18
        self.zoom_step = 0.1
        self.min_zoom = 0.3
        self.max_zoom = 50.0
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        # 投影转换
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xmin, ymax = self.transformer.transform(self.LON_MIN, self.LAT_MAX)
        xmax, ymin = self.transformer.transform(self.LON_MAX, self.LAT_MIN)
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

        self.use_projection = False
        self.fps = fps

        # 阵营底圆使用固定屏幕尺寸，缓存透明贴图供每帧复用。
        self._side_backgrounds = {}
        # 巡航姿态需要细粒度航向，使用有界LRU避免长局累计数千张旋转贴图。
        self._model_cache = OrderedDict()
        self._model_cache_limit = 2048
        self._hive_sprite_sources = {}
        self._hive_sprite_cache = OrderedDict()
        self._hive_sprite_cache_limit = 24
        # 地面、网格、部署区、标题和图例在相机不变时均为静态图层。
        self._static_layer_cache = OrderedDict()
        self._static_layer_cache_limit = 6

        # ========== 鼠标位置 ==========
        self.mouse_x = 0
        self.mouse_y = 0

        self.radar_angle = 0  # 视野塔角度

        self.draw_radius = 8

        # ========== 3D透视视图 ==========
        # Pygame保持轻量级软件渲染，通过轴测投影表达真实经纬度和高度。
        self.view_3d = True
        # 保留东西向间距，让红方在福建侧、蓝方在台湾侧，中间跨越海峡。
        self.camera_yaw_deg = -25.0
        self.camera_pitch_deg = 30.0
        self.altitude_pixels_per_km = 28.0
        self.show_altitude_labels = False
        self.show_satellite = False
        self._entity_screen_positions = {}
        self._last_entity_positions = {}
        self._motion_directions = {}
        self._motion_altitude_deltas = {}
        self._smoothed_motion_directions = {}
        self._smoothed_visual_positions = {}
        self._smoothed_position_deltas = {}
        self._stage_counts = {}
        self._last_health = {}
        self._last_entity_types = {}
        self._last_effect_positions = {}
        self._display_effect_positions = {}
        # M01's raw launch longitudes place H closest to Taiwan and L farthest
        # inland.  The geographic UI presents the requested tactical order at
        # launch (H rear, M middle, L front), then blends back to the actual
        # coordinates once a vehicle starts travelling.
        self._geographic_initial_positions = {}
        self._geographic_initial_type_ranges = {}
        self._explosions = {}
        self._craters = {}
        self._communication_edge_count = 0

    @classmethod
    def render(cls, state: dict) -> bytes:
        """将组织方物理态势绘制为单帧二维 PNG。"""
        # 控制节点保留在系统状态中，地图只绘制行动和感知实体。
        entities = {
            identifier: dict(data, nameChn=str(data["id"]))
            for identifier, data in state["entities"].items()
            if data["type"] != 35000
        }
        area, deployment = state["map_area"], state["deployment_area"]
        points = [(area["lonMin"], area["latMin"]), (area["lonMax"], area["latMax"])]
        points.extend((data["position"]["lon"], data["position"]["lat"]) for data in entities.values())
        points.extend(tuple(point) for polygon in deployment["coordinates"] for point in polygon)
        if any(not math.isfinite(axis) for point in points for axis in point):
            raise ValueError("Render coordinates must be finite")

        # 按地图、部署区和实体位置取景，并留出图标边距与图例侧栏。
        width, height, margin, legend_width = 1000, 800, 80, 180
        west, east = min(p[0] for p in points), max(p[0] for p in points)
        south, north = min(p[1] for p in points), max(p[1] for p in points)
        scale = max((east - west) / (width - 2 * margin),
                    (north - south) / (height - 2 * margin), 1e-6)
        lon, lat = (west + east) / 2, (south + north) / 2
        renderer = cls(lon - width * scale / 2, lon + width * scale / 2,
                       lat - height * scale / 2, lat + height * scale / 2)
        # 组织方静态预览维持原有二维协议；实时human窗口默认显示三维高度。
        renderer.view_3d = False
        renderer.show_satellite = True
        renderer.zoom = 1.0
        fonts_initialized = pygame.font.get_init()
        try:
            if not fonts_initialized:
                pygame.font.init()
            renderer.font = pygame.font.Font(None, 16)
            renderer.font_legend = pygame.font.Font(None, 20)
            surface = pygame.Surface((width + legend_width, height))
            surface.fill(renderer.BACKGROUND_COLOR)
            renderer._draw_coordinate_system(surface)
            frame = ((area["lonMin"], area["latMin"]), (area["lonMax"], area["latMin"]),
                     (area["lonMax"], area["latMax"]), (area["lonMin"], area["latMax"]))
            pygame.draw.polygon(surface, renderer.MUTED_TEXT_COLOR,
                                [renderer._geo_to_screen_linear(*point) for point in frame], 1)
            for polygon in deployment["coordinates"]:
                pygame.draw.polygon(surface, renderer.SIDE_COLORS[0],
                                    [renderer._geo_to_screen_linear(*point) for point in polygon], 2)
            renderer._draw_entities(surface, entities, show_all=True)
            renderer.draw_legend2(surface, labels={
                "毁灭者": "Destroyer", "突击者": "Assault", "清道夫": "Scavenger", "天眼": "Satellite",
                "母巢": "Hive", "哨卫巢": "Sentry", "暗巢": "Hidden hive", "自爆虫群": "Interceptors",
            }, side_labels={"人族": "Red", "虫族": "Blue"})
            output = BytesIO()
            pygame.image.save(surface, output, "preview.png")
            return output.getvalue()
        finally:
            if not fonts_initialized:
                pygame.font.quit()

    def start(self):
        """启动渲染线程，并等待X11窗口完成首帧映射。"""
        self._ready_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._render_loop, daemon=True)
        self.thread.start()
        # 等待会释放GIL，避免原生仿真主循环抢占导致VNC窗口迟迟不出现。
        if not self._ready_event.wait(timeout=10.0):
            self.running = False
            raise RuntimeError("渲染窗口在10秒内未完成初始化")

    def stop(self):
        """停止渲染线程"""
        with self._frame_condition:
            self.running = False
            self._frame_condition.notify_all()
        if self.thread:
            self.thread.join()

    def update_data(self, entities: dict, *, step=None, wait=False):
        """
        提交一帧态势；human模式可等待本帧真正显示后再推进下一仿真步。
        :param entities: {entity_id: {"position": {"lon": , "lat": , "alt": }}}
        :param step: 该帧对应的仿真步数，显示于HUD
        :param wait: 是否等待渲染线程完成本帧
        """
        # 连续帧位置差比部分原生模型未更新的velNue更可靠，用于巡航姿态航向。
        current_positions = {}
        current_health = {}
        current_entity_types = {}
        effect_time = time.perf_counter()
        for identifier, data in entities.items():
            position = data.get("position", {})
            try:
                lon = float(position.get("lon", 0.0))
                lat = float(position.get("lat", 0.0))
                alt = float(position.get("alt", 0.0))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (lon, lat, alt)):
                continue
            current_positions[identifier] = (lon, lat, alt)
            try:
                health = float(data.get("health", 0.0))
            except (TypeError, ValueError):
                health = 0.0
            current_health[identifier] = health
            try:
                entity_type = int(data.get("type", -1))
            except (TypeError, ValueError):
                entity_type = -1
            current_entity_types[identifier] = entity_type
            if health > 0.0 and entity_type in self.HIVE_SPRITE_FILES:
                self._craters.pop(identifier, None)
            previous_health = self._last_health.get(identifier)
            if previous_health is not None and previous_health > 0.0 >= health:
                effect_position = self._last_effect_positions.get(
                    identifier, (lon, lat, alt),
                )
                self._explosions[identifier] = (effect_position, effect_time)
                if entity_type in self.HIVE_SPRITE_FILES:
                    self._craters[identifier] = (effect_position, entity_type)
            self._last_effect_positions[identifier] = (lon, lat, alt)
            previous = self._last_entity_positions.get(identifier)
            if previous is None:
                continue
            delta_lon, delta_lat = lon - previous[0], lat - previous[1]
            delta_alt = alt - previous[2]
            center_lat = (lat + previous[1]) / 2.0
            east_km = delta_lon * 111.32 * math.cos(math.radians(center_lat))
            north_km = delta_lat * 111.32
            if math.hypot(east_km, north_km) > 1e-7:
                raw_direction = (north_km, east_km)
                self._motion_directions[identifier] = raw_direction
                try:
                    stage = int(float(data.get("stage", -1)))
                except (TypeError, ValueError):
                    stage = -1
                if data.get("type") in (21000, 21001) and stage >= 3:
                    old_direction = self._smoothed_motion_directions.get(identifier)
                    if old_direction is None:
                        smooth_direction = raw_direction
                    else:
                        # 低通航向只作用于显示，抑制原生坐标微抖；真实位置不变。
                        alpha = 0.16
                        smooth_direction = (
                            old_direction[0] * (1.0 - alpha) + north_km * alpha,
                            old_direction[1] * (1.0 - alpha) + east_km * alpha,
                        )
                    if math.hypot(*smooth_direction) > 1e-9:
                        self._smoothed_motion_directions[identifier] = smooth_direction
            try:
                stage = int(float(data.get("stage", -1)))
            except (TypeError, ValueError):
                stage = -1
            if data.get("type") in (21000, 21001) and stage >= 3:
                raw_delta = (delta_lon, delta_lat, delta_alt)
                old_delta = self._smoothed_position_deltas.get(identifier)
                visual_previous = self._smoothed_visual_positions.get(identifier)
                if old_delta is None or visual_previous is None:
                    smooth_delta = raw_delta
                    visual_position = (lon, lat, alt)
                else:
                    velocity_alpha = 0.24
                    smooth_delta = tuple(
                        old * (1.0 - velocity_alpha) + raw * velocity_alpha
                        for old, raw in zip(old_delta, raw_delta)
                    )
                    predicted = tuple(
                        value + delta
                        for value, delta in zip(visual_previous, smooth_delta)
                    )
                    correction_gain = 0.28
                    visual_position = tuple(
                        estimate + (actual - estimate) * correction_gain
                        for estimate, actual in zip(predicted, (lon, lat, alt))
                    )
                    error_km = math.hypot(
                        (visual_position[0] - lon) * 111.32
                        * math.cos(math.radians(lat)),
                        (visual_position[1] - lat) * 111.32,
                    )
                    if error_km > 5.0:
                        visual_position = (lon, lat, alt)
                        smooth_delta = raw_delta
                self._smoothed_position_deltas[identifier] = smooth_delta
                self._smoothed_visual_positions[identifier] = visual_position
            else:
                self._smoothed_visual_positions.pop(identifier, None)
                self._smoothed_position_deltas.pop(identifier, None)
            altitude_delta = delta_alt
            if abs(altitude_delta) > 1e-6:
                self._motion_altitude_deltas[identifier] = altitude_delta
        # 从态势表直接消失也视为一次击毁/退出特效，使用最后已知位置。
        for identifier, health in self._last_health.items():
            if health > 0.0 and identifier not in current_health:
                position = self._last_effect_positions.get(identifier)
                if position is not None:
                    self._explosions[identifier] = (position, effect_time)
                    entity_type = self._last_entity_types.get(identifier, -1)
                    if entity_type in self.HIVE_SPRITE_FILES:
                        self._craters[identifier] = (position, entity_type)
        self._last_health = current_health
        self._last_entity_types = current_entity_types
        self._last_entity_positions = current_positions
        self._motion_directions = {
            identifier: direction
            for identifier, direction in self._motion_directions.items()
            if identifier in current_positions
        }
        self._motion_altitude_deltas = {
            identifier: delta
            for identifier, delta in self._motion_altitude_deltas.items()
            if identifier in current_positions
        }
        self._smoothed_motion_directions = {
            identifier: direction
            for identifier, direction in self._smoothed_motion_directions.items()
            if identifier in current_positions
        }
        self._smoothed_visual_positions = {
            identifier: position
            for identifier, position in self._smoothed_visual_positions.items()
            if identifier in current_positions
        }
        self._smoothed_position_deltas = {
            identifier: delta
            for identifier, delta in self._smoothed_position_deltas.items()
            if identifier in current_positions
        }
        submitted_at = time.perf_counter()
        if self._last_submit_time is not None:
            interval_ms = (submitted_at - self._last_submit_time) * 1000.0
            self._step_interval_ms = (
                interval_ms if self._step_interval_ms <= 0.0
                else self._step_interval_ms * 0.90 + interval_ms * 0.10
            )
        self._last_submit_time = submitted_at
        display_entities = {}
        for identifier, data in entities.items():
            visual_position = self._smoothed_visual_positions.get(identifier)
            if visual_position is None:
                display_entities[identifier] = data
                continue
            displayed = data.copy()
            displayed["position"] = dict(
                data.get("position", {}),
                lon=visual_position[0], lat=visual_position[1], alt=visual_position[2],
            )
            display_entities[identifier] = displayed
        with self._frame_condition:
            self.entities_data = display_entities
            self._submitted_frame += 1
            sequence = self._submitted_frame
            self._pending_step = self.current_step if step is None else int(step)
            self._frame_condition.notify_all()
            if wait:
                if not self.running:
                    raise RuntimeError("渲染器未运行，无法确认态势帧")
                completed = self._frame_condition.wait_for(
                    lambda: self._rendered_frame >= sequence or not self.running,
                    timeout=30.0,
                )
                if not completed:
                    raise RuntimeError(
                        f"仿真第 {self._pending_step} 步的UI帧在30秒内未完成"
                    )
                if not self.running and self._rendered_frame < sequence:
                    raise RuntimeError("渲染窗口已关闭，态势帧未完成")

    def update_area(self, red_area: list[list[list[float]]]):
        """
        更新区域数据（非阻塞，只保留最新）
        :param red_area: 红方区域
        """
        self.red_area = red_area[0] if red_area else []

    def _motion_extrapolation(self, entities, now):
        """在两个真实仿真步之间生成只用于显示的匀速中间态。

        返回 ``(entities, phase, moving)``。输入状态和仿真器均不会被修改；
        新真实步到来时必定以原始快照覆盖该显示预测。
        """
        if self._last_submit_time is None or self._step_interval_ms <= 0.0:
            return entities, 0.0, False
        expected_wall_seconds = max(1e-3, self._step_interval_ms / 1000.0)
        phase = max(0.0, min(0.95, (now - self._last_submit_time) / expected_wall_seconds))
        if phase <= 0.0:
            return entities, 0.0, False

        predicted = {}
        moving = False
        for identifier, data in entities.items():
            try:
                stage = int(float(data.get("stage", -1)))
            except (TypeError, ValueError):
                stage = -1
            velocity = data.get("velocity_nue", {})
            position = data.get("position", {})
            try:
                north = float(velocity.get("x", 0.0))
                up = float(velocity.get("y", 0.0))
                east = float(velocity.get("z", 0.0))
                lon = float(position.get("lon", 0.0))
                lat = float(position.get("lat", 0.0))
                alt = float(position.get("alt", 0.0))
            except (AttributeError, TypeError, ValueError):
                predicted[identifier] = data
                continue
            exact_motion = self._motion_directions.get(identifier)
            altitude_delta = self._motion_altitude_deltas.get(identifier, 0.0)
            speed = math.sqrt(north * north + up * up + east * east)
            # 只补间移动阶段；Launch/Rise保持引擎给出的精确姿态。
            if stage not in (3, 4, 5) or (
                exact_motion is None and speed <= 1e-3 and abs(altitude_delta) <= 1e-6
            ):
                predicted[identifier] = data
                continue
            if exact_motion is not None:
                north_delta_m = exact_motion[0] * 1000.0
                east_delta_m = exact_motion[1] * 1000.0
                up_delta_m = altitude_delta
            else:
                north_delta_m = north * self.sim_step_seconds
                east_delta_m = east * self.sim_step_seconds
                up_delta_m = up * self.sim_step_seconds
            meters_per_degree_lon = max(
                1e-6, 111_320.0 * math.cos(math.radians(lat)),
            )
            displayed = data.copy()
            displayed["position"] = dict(
                position,
                lon=lon + east_delta_m * phase / meters_per_degree_lon,
                lat=lat + north_delta_m * phase / 111_320.0,
                alt=max(0.0, alt + up_delta_m * phase),
            )
            predicted[identifier] = displayed
            moving = True
        return predicted, phase, moving

    def _init_lonlat_pixel_ratio(self):
        """计算每像素对应的横纵坐标跨度（正方形像素，保持不变形）。
        根据初始屏幕尺寸与场景坐标范围确定比例，并记录地图中心，
        后续不再改变该比例。
        """
        lon_lat_pixel_ratio = 1  # 横纵坐标使用相同比例

        viewport_width, viewport_height = self._coordinate_viewport_size()
        self.lat_per_pixel = (self.LAT_MAX - self.LAT_MIN) / viewport_height
        self.lon_per_pixel = self.lat_per_pixel * lon_lat_pixel_ratio
        lon_pixel = (self.LON_MAX - self.LON_MIN) / self.lon_per_pixel
        if lon_pixel > viewport_width:
            self.lon_per_pixel = (self.LON_MAX - self.LON_MIN) / viewport_width
            self.lat_per_pixel = self.lon_per_pixel / lon_lat_pixel_ratio

        # 记录地图中心（恒定，不随窗口缩放变化）
        self.geo_center_lon = (self.LON_MAX + self.LON_MIN) / 2
        self.geo_center_lat = (self.LAT_MAX + self.LAT_MIN) / 2

    def _recalculate_lonlat_boundary(self):
        """根据屏幕尺寸重新计算坐标范围（仅重新居中，不改变每像素比例）。
        使用初始化时确定的恒定像素比例与地图中心，
        将地图中心对齐到当前窗口中心，保证窗口缩放时坐标轴（位于地图中心）
        始终居中、不发生偏移。
        """
        viewport_width, viewport_height = self._coordinate_viewport_size()
        self.LON_MIN = self.geo_center_lon - self.lon_per_pixel * viewport_width / 2
        self.LON_MAX = self.geo_center_lon + self.lon_per_pixel * viewport_width / 2
        self.LAT_MIN = self.geo_center_lat - self.lat_per_pixel * viewport_height / 2
        self.LAT_MAX = self.geo_center_lat + self.lat_per_pixel * viewport_height / 2

    def _coordinate_viewport_size(self):
        """返回旋转前的二维坐标画布尺寸，90度时交换宽高。"""
        angle = self.view_rotation_deg % 180.0
        if abs(angle - 90.0) < 1e-6:
            return float(self.MAP_H), float(self.MAP_W)
        return float(self.MAP_W), float(self.MAP_H)

    def _coordinate_world_center(self):
        width, height = self._coordinate_viewport_size()
        return width / 2.0, height / 2.0

    def _world_to_screen_float(self, world_x, world_y):
        """把未旋转的二维世界坐标映射到当前视口。"""
        world_cx, world_cy = self._coordinate_world_center()
        screen_cx, screen_cy = self.MAP_W / 2.0, self.MAP_H / 2.0
        angle = math.radians(self.view_rotation_deg)
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        dx, dy = world_x - world_cx, world_y - world_cy
        rotated_x = cos_angle * dx + sin_angle * dy
        rotated_y = -sin_angle * dx + cos_angle * dy
        return (
            rotated_x * self.zoom + screen_cx + self.offset_x,
            rotated_y * self.zoom + screen_cy + self.offset_y,
        )

    # ========== 缩放控制方法 ==========
    def zoom_in(self):
        """放大"""
        self.zoom = min(self.zoom + self.zoom_step, self.max_zoom)

    def zoom_out(self):
        """缩小"""
        self.zoom = max(self.zoom - self.zoom_step, self.min_zoom)

    def reset_view(self):
        """重置视图"""
        if self.geographic_layout == "fujian_taiwan":
            self.view_3d = False
            self.view_rotation_deg = 0.0
            self.zoom = 1.0
            self.offset_x = 0.0
            self.offset_y = 0.0
            return
        self.zoom = 1.35
        self.offset_x = self.MAP_W * 0.11
        self.offset_y = 0
        self.camera_yaw_deg = -25.0
        self.camera_pitch_deg = 30.0
        self.altitude_pixels_per_km = 28.0

    def rotate_camera(self, yaw_delta=0.0, pitch_delta=0.0):
        """旋转3D相机；俯仰限制在接近侧视和接近俯视之间。"""
        self.camera_yaw_deg = (self.camera_yaw_deg + yaw_delta) % 360.0
        self.camera_pitch_deg = max(12.0, min(82.0, self.camera_pitch_deg + pitch_delta))

    def change_altitude_scale(self, factor):
        """调整高度显示比例，不改变仿真中的实际高度。"""
        self.altitude_pixels_per_km = max(
            4.0, min(120.0, self.altitude_pixels_per_km * factor)
        )

    def _zoom_at_mouse(self, zoom_func):
        """在鼠标位置执行缩放"""
        mouse_x, mouse_y = pygame.mouse.get_pos()
        cx, cy = self.MAP_W // 2, self.MAP_H // 2

        dx = mouse_x - cx - self.offset_x
        dy = mouse_y - cy - self.offset_y
        world_at_mouse_x = dx / self.zoom if self.zoom != 0 else 0
        world_at_mouse_y = dy / self.zoom if self.zoom != 0 else 0

        zoom_func()  # 执行放大或缩小

        new_dx = world_at_mouse_x * self.zoom
        new_dy = world_at_mouse_y * self.zoom
        self.offset_x = mouse_x - cx - new_dx
        self.offset_y = mouse_y - cy - new_dy

    def _geo_to_screen_linear(self, lon, lat):
        """
        线性映射（带缩放和偏移）
        返回屏幕坐标，原点在屏幕中心
        """
        sx, sy = self._geo_to_screen_linear_float(lon, lat)
        return int(round(sx)), int(round(sy))

    def _geo_to_screen_linear_float(self, lon, lat):
        """线性映射的浮点版本，供移动实体保留亚像素位移。"""
        # 将场景坐标映射到世界坐标 (0~MAP_W, 0~MAP_H)
        wx = (lon - self.LON_MIN) / self.lon_per_pixel # 距离屏幕左边缘的像素数
        wy = (self.LAT_MAX - lat) / self.lat_per_pixel # 距离屏幕上边缘的像素数

        return self._world_to_screen_float(wx, wy)

    def _geo_to_screen_projection(self, lon, lat):
        """
        投影转换（带缩放和偏移）
        返回屏幕坐标，原点在屏幕中心
        """
        x, y = self.transformer.transform(lon, lat)
        viewport_width, viewport_height = self._coordinate_viewport_size()
        wx = (x - self.xmin) / (self.xmax - self.xmin) * viewport_width
        wy = (self.ymax - y) / (self.ymax - self.ymin) * viewport_height
        sx, sy = self._world_to_screen_float(wx, wy)
        return int(round(sx)), int(round(sy))

    def _world_to_screen(self, world_x, world_y):
        """
        将世界坐标（0~MAP_W, 0~MAP_H）转换为屏幕坐标（带缩放和偏移）
        原点在屏幕中心
        """
        sx, sy = self._world_to_screen_float(world_x, world_y)
        return int(round(sx)), int(round(sy))

    def _screen_to_world(self, screen_x, screen_y):
        """
        将屏幕坐标转换为世界坐标（0~MAP_W, 0~MAP_H）
        """
        if self.zoom == 0:
            return 0.0, 0.0
        screen_cx, screen_cy = self.MAP_W / 2.0, self.MAP_H / 2.0
        rotated_x = (screen_x - screen_cx - self.offset_x) / self.zoom
        rotated_y = (screen_y - screen_cy - self.offset_y) / self.zoom
        angle = math.radians(self.view_rotation_deg)
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        dx = cos_angle * rotated_x - sin_angle * rotated_y
        dy = sin_angle * rotated_x + cos_angle * rotated_y
        world_cx, world_cy = self._coordinate_world_center()
        return dx + world_cx, dy + world_cy

    def _screen_to_geo_linear(self, screen_x, screen_y):
        """
        将屏幕坐标转换为场景坐标（线性映射）
        """
        wx, wy = self._screen_to_world(screen_x, screen_y)
        lon = wx * self.lon_per_pixel + self.LON_MIN
        lat = self.LAT_MAX - wy * self.lat_per_pixel
        return lon, lat

    def _scene_metrics(self):
        """返回场景中心、东西/南北公里跨度和当前地面像素比例。"""
        center_lon = (self.SCENE_LON_MIN + self.SCENE_LON_MAX) / 2.0
        center_lat = (self.SCENE_LAT_MIN + self.SCENE_LAT_MAX) / 2.0
        km_per_deg_lat = 111.32
        km_per_deg_lon = max(1e-6, 111.32 * math.cos(math.radians(center_lat)))
        width_km = max(1e-6, (self.SCENE_LON_MAX - self.SCENE_LON_MIN) * km_per_deg_lon)
        height_km = max(1e-6, (self.SCENE_LAT_MAX - self.SCENE_LAT_MIN) * km_per_deg_lat)
        # 预留右侧图例和上方高度空间；zoom只影响显示，不改变物理比例。
        pixels_per_km = min(
            self.MAP_W * 0.70 / width_km,
            self.MAP_H * 0.60 / height_km,
        ) * self.zoom
        return center_lon, center_lat, km_per_deg_lon, km_per_deg_lat, pixels_per_km

    def _geo_to_screen_3d(self, lon, lat, alt_m=0.0):
        """将经纬度与海拔米转换为轴测投影屏幕坐标。"""
        x, y = self._geo_to_screen_3d_float(lon, lat, alt_m)
        return int(round(x)), int(round(y))

    def _geo_to_screen_3d_float(self, lon, lat, alt_m=0.0):
        """轴测投影的浮点版本，避免巡航小位移在投影入口被截断。"""
        center_lon, center_lat, km_lon, km_lat, scale = self._scene_metrics()
        east_km = (float(lon) - center_lon) * km_lon
        north_km = (float(lat) - center_lat) * km_lat
        altitude_km = max(0.0, float(alt_m)) / 1000.0

        yaw = math.radians(self.camera_yaw_deg)
        pitch = math.radians(self.camera_pitch_deg)
        lateral_km = east_km * math.cos(yaw) - north_km * math.sin(yaw)
        depth_km = east_km * math.sin(yaw) + north_km * math.cos(yaw)

        x = self.MAP_W * 0.46 + self.offset_x + lateral_km * scale
        y = (
            self.MAP_H * 0.58 + self.offset_y
            + depth_km * scale * math.sin(pitch)
            - altitude_km * self.altitude_pixels_per_km * self.zoom * math.cos(pitch)
        )
        return x, y

    def _camera_depth(self, lon, lat):
        """返回地面点在相机方向的深度，用于从远到近绘制实体。"""
        center_lon, center_lat, km_lon, km_lat, _ = self._scene_metrics()
        east_km = (float(lon) - center_lon) * km_lon
        north_km = (float(lat) - center_lat) * km_lat
        yaw = math.radians(self.camera_yaw_deg)
        return east_km * math.sin(yaw) + north_km * math.cos(yaw)

    def _draw_3d_floor(self, screen, divisions=10):
        """绘制有限地图地面网格、边界和E/N/Z三轴。"""
        lon0, lon1 = self.SCENE_LON_MIN, self.SCENE_LON_MAX
        lat0, lat1 = self.SCENE_LAT_MIN, self.SCENE_LAT_MAX

        # 地面经纬网：网格点始终位于零高度平面。
        for index in range(divisions + 1):
            ratio = index / divisions
            lon = lon0 + (lon1 - lon0) * ratio
            lat = lat0 + (lat1 - lat0) * ratio
            pygame.draw.line(
                screen, self.GRID_COLOR,
                self._geo_to_screen_3d(lon, lat0, 0),
                self._geo_to_screen_3d(lon, lat1, 0), 1,
            )
            pygame.draw.line(
                screen, self.GRID_COLOR,
                self._geo_to_screen_3d(lon0, lat, 0),
                self._geo_to_screen_3d(lon1, lat, 0), 1,
            )

        corners = [
            self._geo_to_screen_3d(lon0, lat0, 0),
            self._geo_to_screen_3d(lon1, lat0, 0),
            self._geo_to_screen_3d(lon1, lat1, 0),
            self._geo_to_screen_3d(lon0, lat1, 0),
        ]
        pygame.draw.polygon(screen, self.MUTED_TEXT_COLOR, corners, 2)

        origin = self._geo_to_screen_3d(lon0, lat0, 0)
        east = self._geo_to_screen_3d(lon1, lat0, 0)
        north = self._geo_to_screen_3d(lon0, lat1, 0)
        top = self._geo_to_screen_3d(lon0, lat0, 12000)
        pygame.draw.line(screen, (160, 88, 100), origin, east, 3)
        pygame.draw.line(screen, (81, 104, 154), origin, north, 3)
        pygame.draw.line(screen, (60, 126, 84), origin, top, 3)
        for point, label, color in (
            (east, "E", (160, 88, 100)),
            (north, "N", (81, 104, 154)),
            (top, "Z 12 km", (60, 126, 84)),
        ):
            text = self.font.render(label, True, color)
            screen.blit(text, (point[0] + 5, point[1] - 12))
        for altitude_m in (3000, 6000, 9000):
            point = self._geo_to_screen_3d(lon0, lat0, altitude_m)
            pygame.draw.line(screen, (60, 126, 84),
                             (point[0] - 4, point[1]), (point[0] + 4, point[1]), 1)
            text = self.font.render(f"{altitude_m // 1000}", True, self.MUTED_TEXT_COLOR)
            screen.blit(text, (point[0] - 18, point[1] - 6))

    def _draw_3d_range_ring(self, screen, lon, lat, range_km, color):
        """Draw a deliberately non-metric circular detection indicator."""
        del range_km
        center = self._geo_to_screen_3d(lon, lat, 0)
        pygame.draw.circle(
            screen, color, (round(center[0]), round(center[1])),
            int(self.range_ring_radius_px), 1,
        )

    def _draw_2d_range_ring(self, screen, lon, lat, range_km, color):
        """Draw a deliberately non-metric circular detection indicator."""
        del range_km
        center = self._geo_to_screen_linear_float(lon, lat)
        pygame.draw.circle(
            screen, color, (round(center[0]), round(center[1])),
            int(self.range_ring_radius_px), 1,
        )

    def _draw_view_hud(self, screen):
        """显示当前视图模式、相机参数与交互快捷键。"""
        frame_label = (
            "SYNCED" if self._display_phase <= 0.0
            else f"INTERP {self._display_phase:.2f}"
        )
        if self.view_3d:
            lines = [
                f"STEP {self.current_step}  {frame_label}",
                f"3D  yaw {self.camera_yaw_deg:.0f}  pitch {self.camera_pitch_deg:.0f}",
                f"height {self.altitude_pixels_per_km:.0f} px/km  zoom {self.zoom:.1f}x",
                f"UI {self._present_rate_hz:.1f} fps  draw {self._render_time_ms:.1f} ms  step {self._step_interval_ms:.1f} ms",
                f"COMM direct links {self._communication_edge_count}",
                "stages  " + "  ".join(
                    f"{self.STAGE_LABELS.get(stage, stage)}:{self._stage_counts.get(stage, 0)}"
                    for stage in (1, 2, 3, 4)
                ),
                "V:2D/3D  arrows:camera  [ ]:height  H:labels  R:reset",
            ]
        else:
            lines = [
                f"STEP {self.current_step}  {frame_label}",
                f"2D  zoom {self.zoom:.1f}x",
                f"UI {self._present_rate_hz:.1f} fps  draw {self._render_time_ms:.1f} ms  step {self._step_interval_ms:.1f} ms",
                f"COMM direct links {self._communication_edge_count}",
                (
                    "fixed geographic view  R:reset"
                    if self.geographic_layout == "fujian_taiwan"
                    else "V:2D/3D  wheel:zoom  drag:pan  R:reset"
                ),
            ]
        width = max(self.font.size(line)[0] for line in lines) + 16
        panel = pygame.Surface((width, len(lines) * 18 + 10), pygame.SRCALPHA)
        panel.fill((*self.PANEL_COLOR, 225))
        pygame.draw.rect(panel, self.PANEL_BORDER_COLOR, panel.get_rect(), 1)
        for index, line in enumerate(lines):
            panel.blit(self.font.render(line, True, self.TEXT_COLOR), (8, 6 + 18 * index))
        screen.blit(panel, (10, 10))

    def _draw_coordinate_system(self, screen):
        """
        绘制无限延伸的坐标系（与背景绑定）
        """
        cx, cy = self._coordinate_world_center()

        # 计算网格线间距（随着缩放自动调整）
        base_spacing = 50
        grid_spacing = max(20, base_spacing * self.zoom)

        # 计算需要绘制的网格范围
        corners_world = [
            self._screen_to_world(0, 0),
            self._screen_to_world(self.MAP_W, 0),
            self._screen_to_world(self.MAP_W, self.MAP_H),
            self._screen_to_world(0, self.MAP_H)
        ]

        world_xs = [p[0] for p in corners_world]
        world_ys = [p[1] for p in corners_world]
        min_wx, max_wx = min(world_xs), max(world_xs)
        min_wy, max_wy = min(world_ys), max(world_ys)

        extend = grid_spacing / self.zoom * 2 if self.zoom != 0 else 100
        min_wx -= extend
        max_wx += extend
        min_wy -= extend
        max_wy += extend

        grid_spacing_world = grid_spacing / self.zoom if self.zoom != 0 else grid_spacing
        start_wx = (min_wx // grid_spacing_world) * grid_spacing_world
        start_wy = (min_wy // grid_spacing_world) * grid_spacing_world

        # ========== 绘制网格线（无限延伸） ==========
        grid_color = self.GRID_COLOR

        x = start_wx
        while x <= max_wx:
            p1 = self._world_to_screen(x, min_wy)
            p2 = self._world_to_screen(x, max_wy)
            pygame.draw.line(screen, grid_color, p1, p2, 1)
            x += grid_spacing_world

        y = start_wy
        while y <= max_wy:
            p1 = self._world_to_screen(min_wx, y)
            p2 = self._world_to_screen(max_wx, y)
            pygame.draw.line(screen, grid_color, p1, p2, 1)
            y += grid_spacing_world

        # ========== 绘制坐标轴（无限延伸） ==========
        # X轴（红色）- 水平方向
        axis_color = (160, 88, 100)
        p1 = self._world_to_screen(min_wx, cy)
        p2 = self._world_to_screen(max_wx, cy)
        pygame.draw.line(screen, axis_color, p1, p2, 2)

        x_axis_color = axis_color

        # Y轴（蓝色）- 垂直方向
        axis_color = (81, 104, 154)
        p1 = self._world_to_screen(cx, min_wy)
        p2 = self._world_to_screen(cx, max_wy)
        pygame.draw.line(screen, axis_color, p1, p2, 2)

        y_axis_color = axis_color

        # ========== 绘制原点标记 ==========
        origin_screen = self._world_to_screen(cx, cy)
        if -20 <= origin_screen[0] <= self.MAP_W + 20 and -20 <= origin_screen[1] <= self.MAP_H + 20:
            pygame.draw.circle(screen, self.MUTED_TEXT_COLOR, origin_screen, 8, 2)
            pygame.draw.circle(screen, self.MUTED_TEXT_COLOR, origin_screen, 3)
            origin_text = self.font.render(f"O", True, self.MUTED_TEXT_COLOR)
            screen.blit(origin_text, (origin_screen[0] + 12, origin_screen[1] - 8))

        # ========== 显示X轴和Y轴箭头及标签（跟随旋转后的坐标系） ==========
        x_label = self.font.render("X", True, (160, 88, 100))
        y_label = self.font.render("Y", True, (81, 104, 154))
        angle = math.radians(self.view_rotation_deg)
        x_direction = (math.cos(angle), -math.sin(angle))
        # 地理Y正方向是纬度增加，对应世界坐标中的-y。
        y_direction = (-math.sin(angle), -math.cos(angle))

        def draw_axis_arrow(direction, color, label):
            origin_x, origin_y = origin_screen
            margin = 7.0
            candidates = []
            if abs(direction[0]) > 1e-9:
                for boundary_x in (margin, self.MAP_W - margin):
                    distance = (boundary_x - origin_x) / direction[0]
                    boundary_y = origin_y + distance * direction[1]
                    if distance > 0 and margin <= boundary_y <= self.MAP_H - margin:
                        candidates.append((distance, boundary_x, boundary_y))
            if abs(direction[1]) > 1e-9:
                for boundary_y in (margin, self.MAP_H - margin):
                    distance = (boundary_y - origin_y) / direction[1]
                    boundary_x = origin_x + distance * direction[0]
                    if distance > 0 and margin <= boundary_x <= self.MAP_W - margin:
                        candidates.append((distance, boundary_x, boundary_y))
            if not candidates:
                return
            _, tip_x, tip_y = min(candidates)
            base_x = tip_x - direction[0] * 11.0
            base_y = tip_y - direction[1] * 11.0
            normal_x, normal_y = -direction[1], direction[0]
            pygame.draw.polygon(screen, color, [
                (round(tip_x), round(tip_y)),
                (round(base_x + normal_x * 5.0), round(base_y + normal_y * 5.0)),
                (round(base_x - normal_x * 5.0), round(base_y - normal_y * 5.0)),
            ])
            label_x = max(2, min(self.MAP_W - label.get_width() - 2,
                                 round(tip_x - direction[0] * 20 - label.get_width() / 2)))
            label_y = max(2, min(self.MAP_H - label.get_height() - 2,
                                 round(tip_y - direction[1] * 20 - label.get_height() / 2)))
            screen.blit(label, (label_x, label_y))

        draw_axis_arrow(x_direction, x_axis_color, x_label)
        draw_axis_arrow(y_direction, y_axis_color, y_label)

        # ========== 绘制轴上的刻度标记（无限延伸） ==========
        # X轴刻度
        step = 100 / self.zoom if self.zoom != 0 else 100
        x_start = (min_wx // step) * step
        x_end = (max_wx // step) * step

        x = x_start
        x_tick_normal = (-x_direction[1], x_direction[0])
        while x <= x_end:
            sx, sy = self._world_to_screen(x, cy)
            if -20 <= sx <= self.MAP_W + 20 and -20 <= sy <= self.MAP_H + 20:
                pygame.draw.line(screen, self.MUTED_TEXT_COLOR,
                                 (sx - x_tick_normal[0] * 5, sy - x_tick_normal[1] * 5),
                                 (sx + x_tick_normal[0] * 5, sy + x_tick_normal[1] * 5), 1)
                if abs(x - cx) > 5:
                    offset = int(x - cx)
                    text = self.font.render(f"{offset}", True, self.MUTED_TEXT_COLOR)
                    screen.blit(text, (sx + x_tick_normal[0] * 8 - text.get_width() / 2,
                                       sy + x_tick_normal[1] * 8 - text.get_height() / 2))
            x += step

        # Y轴刻度
        y_start = (min_wy // step) * step
        y_end = (max_wy // step) * step

        y = y_start
        y_tick_normal = (-y_direction[1], y_direction[0])
        while y <= y_end:
            sx, sy = self._world_to_screen(cx, y)
            if -20 <= sx <= self.MAP_W + 20 and -20 <= sy <= self.MAP_H + 20:
                pygame.draw.line(screen, self.MUTED_TEXT_COLOR,
                                 (sx - y_tick_normal[0] * 5, sy - y_tick_normal[1] * 5),
                                 (sx + y_tick_normal[0] * 5, sy + y_tick_normal[1] * 5), 1)
                if abs(y - cy) > 5:
                    offset = int(cy - y)
                    text = self.font.render(f"{offset}", True, self.MUTED_TEXT_COLOR)
                    screen.blit(text, (sx + y_tick_normal[0] * 8 - text.get_width() / 2,
                                       sy + y_tick_normal[1] * 8 - text.get_height() / 2))
            y += step

    def _draw_side_background(self, surface, x, y, side, radius):
        """在单位中心绘制低透明度阵营底圆，尺寸不随地图缩放。"""
        color = self.SIDE_COLORS.get(side, self.NEUTRAL_COLOR)
        key = (color, radius)
        if key not in self._side_backgrounds:
            diameter = radius * 2 + 1
            disc = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
            pygame.draw.circle(disc, (*color, 45), (radius, radius), radius)
            pygame.draw.circle(disc, (*color, 165), (radius, radius), radius, 1)
            self._side_backgrounds[key] = disc
        surface.blit(
            self._side_backgrounds[key],
            (int(round(x - radius)), int(round(y - radius))),
        )

    @staticmethod
    def _cache_get(cache, key):
        """读取LRU缓存并将命中项移到队尾。"""
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
        return value

    @staticmethod
    def _cache_put(cache, key, value, limit):
        """写入有界LRU缓存，防止长时间UI运行持续增长内存。"""
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    @staticmethod
    def _blit_centered_subpixel(surface, sprite, x, y):
        """按浮点中心绘制贴图；误差扩散让小于1像素的移动逐帧可见。

        Pygame最终仍使用整数像素。这里不在投影入口截断，而是根据中心
        小数部分做确定性的半像素舍入，消除始终向左上截断造成的停顿。
        """
        left = int(round(float(x) - sprite.get_width() / 2.0))
        top = int(round(float(y) - sprite.get_height() / 2.0))
        surface.blit(sprite, (left, top))

    def _scaled_hive_sprite(self, entity_type, size):
        """Load, alpha-crop and cache one realistic hive sprite at display size."""
        cache_key = (int(entity_type), max(1, int(round(size))))
        cached = self._cache_get(self._hive_sprite_cache, cache_key)
        if cached is not None:
            return cached

        if entity_type not in self._hive_sprite_sources:
            filename = self.HIVE_SPRITE_FILES.get(entity_type)
            source = None
            if filename is not None:
                try:
                    loaded = pygame.image.load(str(self.hive_sprite_dir / filename))
                    bounds = loaded.get_bounding_rect(min_alpha=8)
                    if bounds.width > 0 and bounds.height > 0:
                        source = loaded.subsurface(bounds).copy()
                except (OSError, pygame.error):
                    source = None
            self._hive_sprite_sources[entity_type] = source

        source = self._hive_sprite_sources.get(entity_type)
        if source is None:
            return None
        longest = max(source.get_width(), source.get_height())
        target_longest = max(8, int(round(float(size) * 1.1)))
        scale = target_longest / max(1, longest)
        scaled = pygame.transform.smoothscale(
            source,
            (
                max(1, int(round(source.get_width() * scale))),
                max(1, int(round(source.get_height() * scale))),
            ),
        )
        self._cache_put(
            self._hive_sprite_cache, cache_key, scaled,
            self._hive_sprite_cache_limit,
        )
        return scaled

    def _draw_hive_sprite(self, surf, cx, cy, size, entity_type):
        sprite = self._scaled_hive_sprite(entity_type, size)
        if sprite is None:
            return False
        color = self.HIVE_TYPE_COLORS.get(entity_type, self.NEUTRAL_COLOR)
        radius = max(8, int(round(float(size) * 0.58)))
        halo = pygame.Surface((radius * 2 + 5, radius * 2 + 5), pygame.SRCALPHA)
        center = (halo.get_width() // 2, halo.get_height() // 2)
        pygame.draw.circle(halo, (*color, 48), center, radius)
        pygame.draw.circle(halo, (*color, 230), center, radius, 2)
        self._blit_centered_subpixel(surf, halo, cx, cy)
        self._blit_centered_subpixel(surf, sprite, cx, cy)
        return True

    def _platform_screen_heading(self, data):
        """按N-U-E速度计算模型在当前视图中的屏幕航向角。"""
        velocity = data.get("velocity_nue", {})
        position = data.get("position", {})
        try:
            identifier = data.get("id")
            if data.get("type") in (21000, 21001):
                motion = self._smoothed_motion_directions.get(identifier)
            else:
                motion = None
            if motion is None:
                motion = self._motion_directions.get(identifier)
            if motion is None:
                north = float(velocity.get("x", 0.0))
                east = float(velocity.get("z", 0.0))
            else:
                north, east = motion
            lon = float(position.get("lon", 0.0))
            lat = float(position.get("lat", 0.0))
            altitude = float(position.get("alt", 0.0))
        except (TypeError, ValueError):
            return 0.0
        horizontal_speed = math.hypot(north, east)
        if horizontal_speed <= 1e-6:
            return 0.0
        north /= horizontal_speed
        east /= horizontal_speed
        km_per_deg_lat = 111.32
        km_per_deg_lon = max(1e-6, 111.32 * math.cos(math.radians(lat)))
        next_lon = lon + east / km_per_deg_lon
        next_lat = lat + north / km_per_deg_lat
        if self.view_3d:
            origin = self._geo_to_screen_3d_float(lon, lat, altitude)
            target = self._geo_to_screen_3d_float(next_lon, next_lat, altitude)
        else:
            origin = self._geo_to_screen_linear_float(lon, lat)
            target = self._geo_to_screen_linear_float(next_lon, next_lat)
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        if dx == 0 and dy == 0:
            return 0.0
        # 模型原始弹头朝屏幕上方；返回顺时针偏转角。
        return math.degrees(math.atan2(dx, -dy))

    def _draw_vertical_missile_model(self, surface, x, y, size, entity_type, stage, data):
        """绘制Launch/Rise专用竖直侧视模型，尾焰使姿态切换清晰可见。"""
        display_size = float(size)
        supersample = 3
        tilt = 0.0
        if stage == 2:
            heading = self._platform_screen_heading(data)
            tilt = max(-16.0, min(16.0, heading))
        tilt = round(tilt / 4.0) * 4.0
        cache_key = ("vertical-ss3", entity_type, int(display_size), stage, tilt)
        cached = self._cache_get(self._model_cache, cache_key)
        if cached is not None:
            self._blit_centered_subpixel(surface, cached, x, y)
            return

        size = display_size * supersample
        canvas_size = max(38 * supersample, int(size * 2.8))
        marker = pygame.Surface((canvas_size, canvas_size), pygame.SRCALPHA)
        cx = canvas_size / 2.0
        cy = canvas_size / 2.0
        color = self.TYPE_COLORS[entity_type]
        dark = (94, 20, 42) if entity_type == 21000 else (0, 57, 81)
        light = (241, 139, 157) if entity_type == 21000 else (120, 211, 233)
        width = size * (0.20 if entity_type == 21000 else 0.16)
        top = cy - size * 0.48
        bottom = cy + size * 0.31

        # 机身阴影、圆锥弹头和高光共同形成竖直圆柱体。
        body = [
            (cx, top),
            (cx + width, cy - size * 0.24),
            (cx + width, bottom),
            (cx - width, bottom),
            (cx - width, cy - size * 0.24),
        ]
        pygame.draw.polygon(marker, (78, 89, 100, 115),
                            [(px + 2 * supersample, py + 2 * supersample)
                             for px, py in body])
        pygame.draw.polygon(marker, color, body)
        pygame.draw.polygon(marker, dark, [
            (cx, top),
            (cx + width, cy - size * 0.24),
            (cx + width, bottom),
            (cx, bottom),
        ])
        pygame.draw.line(marker, light,
                         (cx - width * 0.38, cy - size * 0.20),
                         (cx - width * 0.38, bottom - 2 * supersample),
                         2 * supersample)

        fin_span = size * (0.34 if entity_type == 21000 else 0.27)
        fins = [
            (cx - width, cy + size * 0.13),
            (cx - fin_span, cy + size * 0.31),
            (cx - width * 0.75, cy + size * 0.25),
            (cx + width * 0.75, cy + size * 0.25),
            (cx + fin_span, cy + size * 0.31),
            (cx + width, cy + size * 0.13),
        ]
        pygame.draw.polygon(marker, dark, fins)

        plume_length = size * (0.42 if stage == 1 else 0.24)
        pygame.draw.polygon(marker, (238, 101, 34), [
            (cx - width * 0.65, bottom),
            (cx, bottom + plume_length),
            (cx + width * 0.65, bottom),
        ])
        pygame.draw.polygon(marker, (255, 219, 87), [
            (cx - width * 0.28, bottom),
            (cx, bottom + plume_length * 0.68),
            (cx + width * 0.28, bottom),
        ])

        rotated_high = pygame.transform.rotate(marker, -tilt)
        rotated = pygame.transform.smoothscale(
            rotated_high,
            (max(1, int(round(rotated_high.get_width() / supersample))),
             max(1, int(round(rotated_high.get_height() / supersample)))),
        )
        self._cache_put(
            self._model_cache, cache_key, rotated, self._model_cache_limit,
        )
        self._blit_centered_subpixel(surface, rotated, x, y)

    def _draw_platform_model(self, surface, x, y, size, drawer, data):
        """按Launch/Rise竖直、Cruise/Terminal水平航向绘制H/M/L。"""
        entity_type = data.get("type")
        try:
            stage = int(float(data.get("stage", -1)))
        except (TypeError, ValueError):
            stage = -1

        # 俯视图中，竖直起飞的H/M只能看到弹头截面；L仍显示旋翼布局。
        if not self.view_3d and stage in (1, 2) and entity_type in (21000, 21001):
            color = self.TYPE_COLORS[entity_type]
            radius = max(3, int(size * 0.22))
            pygame.draw.circle(surface, color, (int(x), int(y)), radius)
            pygame.draw.circle(surface, self.TEXT_COLOR, (int(x), int(y)), radius, 1)
            return

        if self.view_3d and stage in (1, 2) and entity_type in (21000, 21001):
            self._draw_vertical_missile_model(
                surface, x, y, size, entity_type, stage, data,
            )
            return

        # Cruise/Terminal按连续帧地理位移方向贴合水平航迹。
        clockwise = self._platform_screen_heading(data) if stage >= 3 else 0.0
        # 旧版5度量化会造成多步不变、越过阈值后突然跳转。0.1度足以让
        # 每步真实航向变化进入画面；LRU缓存负责限制贴图数量。
        clockwise = round(clockwise, 1)
        supersample = 3 if entity_type in (21000, 21001, 21002) else 1
        cache_key = (
            f"flight-ss{supersample}", entity_type, int(size), stage, clockwise,
        )
        rotated = self._cache_get(self._model_cache, cache_key)
        if rotated is None:
            canvas_size = max(32 * supersample, int(size * 2.4 * supersample))
            marker = pygame.Surface((canvas_size, canvas_size), pygame.SRCALPHA)
            center = canvas_size / 2.0
            drawer(marker, center, center, size * supersample)
            rotated_high = pygame.transform.rotate(marker, -clockwise)
            if supersample > 1:
                rotated = pygame.transform.smoothscale(
                    rotated_high,
                    (max(1, int(round(rotated_high.get_width() / supersample))),
                     max(1, int(round(rotated_high.get_height() / supersample)))),
                )
            else:
                rotated = rotated_high
            self._cache_put(
                self._model_cache, cache_key, rotated, self._model_cache_limit,
            )
        self._blit_centered_subpixel(surface, rotated, x, y)

    def _communication_edges(self, entities, max_degree=2):
        """从真实通信群中筛选直接可达链路，并限制画面拥挤程度。"""
        ranges = {21000: 50_000.0, 21001: 20_000.0, 21002: 10_000.0}
        red = {
            int(data.get("id", identifier)): data
            for identifier, data in entities.items()
            if data.get("side") == 0
            and data.get("type") in ranges
            and data.get("isVisible", False)
            and data.get("health", 0) > 0
        }
        candidates = []
        for identifier, data in red.items():
            peers = set(data.get("comm_peers") or ())
            a_ecf = data.get("pos_ecf", {})
            for peer_id in peers:
                peer_id = int(peer_id)
                if peer_id <= identifier or peer_id not in red:
                    continue
                peer = red[peer_id]
                # 态势图只突出异构协同，避免同型号密集连线淹没实体。
                if peer.get("type") == data.get("type"):
                    continue
                b_ecf = peer.get("pos_ecf", {})
                try:
                    distance_sq = sum(
                        (float(a_ecf[axis]) - float(b_ecf[axis])) ** 2
                        for axis in ("x", "y", "z")
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                limit = min(ranges[data["type"]], ranges[peer["type"]])
                if distance_sq <= limit * limit:
                    candidates.append((distance_sq, identifier, peer_id))
        degree = {identifier: 0 for identifier in red}
        selected = []
        for _, first, second in sorted(candidates):
            if degree[first] >= max_degree or degree[second] >= max_degree:
                continue
            degree[first] += 1
            degree[second] += 1
            selected.append((red[first], red[second]))
        return selected

    def _entity_screen_point(self, data):
        position = data.get("position", {})
        try:
            lon = float(position.get("lon", 0.0))
            lat = float(position.get("lat", 0.0))
            altitude = max(0.0, float(position.get("alt", 0.0)))
        except (TypeError, ValueError):
            return None
        if self.view_3d:
            return self._geo_to_screen_3d_float(lon, lat, altitude)
        return self._geo_to_screen_linear_float(lon, lat)

    @staticmethod
    def _draw_dashed_line(surface, color, start, end, dash=8, gap=6, width=1):
        dx, dy = end[0] - start[0], end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return
        ux, uy = dx / distance, dy / distance
        cursor = 0.0
        while cursor < distance:
            segment_end = min(distance, cursor + dash)
            pygame.draw.line(
                surface, color,
                (start[0] + ux * cursor, start[1] + uy * cursor),
                (start[0] + ux * segment_end, start[1] + uy * segment_end),
                width,
            )
            cursor += dash + gap

    def _draw_communication_links(self, screen, entities):
        """绘制真实直连通信边和沿边移动的数据脉冲。"""
        edges = self._communication_edges(entities)
        self._communication_edge_count = len(edges)
        if not edges:
            return
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        pulse_phase = (time.perf_counter() * 0.55) % 1.0
        for index, (first, second) in enumerate(edges):
            start = self._entity_screen_point(first)
            end = self._entity_screen_point(second)
            if start is None or end is None:
                continue
            self._draw_dashed_line(
                overlay, (31, 226, 246, 180), start, end,
                dash=9, gap=5, width=2,
            )
            phase = (pulse_phase + index * 0.173) % 1.0
            packet = (
                int(round(start[0] + (end[0] - start[0]) * phase)),
                int(round(start[1] + (end[1] - start[1]) * phase)),
            )
            pygame.draw.circle(overlay, (244, 255, 174, 245), packet, 3)
        screen.blit(overlay, (0, 0))

    def _has_active_explosions(self, now=None):
        now = time.perf_counter() if now is None else now
        return any(
            now - started < self.EXPLOSION_DURATION_SECONDS
            for _, started in self._explosions.values()
        )

    def _draw_craters(self, screen):
        """在被摧毁巢穴的地面坐标留下持续弹坑。"""
        if not self._craters:
            return
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        base_sizes = {9400: 25, 9500: 17, 9600: 20}
        for identifier, (position, entity_type) in self._craters.items():
            position = self._display_effect_positions.get(identifier, position)
            lon, lat, _ = position
            if self.view_3d:
                x, y = self._geo_to_screen_3d_float(lon, lat, 0.0)
                width = base_sizes.get(entity_type, 18)
                height = max(6, int(round(width * 0.42)))
            else:
                x, y = self._geo_to_screen_linear_float(lon, lat)
                width = base_sizes.get(entity_type, 18)
                height = max(8, int(round(width * 0.68)))
            center = (int(round(x)), int(round(y)))
            outer = pygame.Rect(0, 0, width, height)
            outer.center = center
            pygame.draw.ellipse(overlay, (35, 29, 23, 165), outer)
            inner = outer.inflate(-max(4, width // 4), -max(3, height // 3))
            pygame.draw.ellipse(overlay, (11, 13, 12, 205), inner)
            pygame.draw.arc(overlay, (107, 78, 48, 190), outer,
                            math.radians(195), math.radians(345), 2)
            for ray_index in range(5):
                angle = math.radians((identifier * 17 + ray_index * 71) % 360)
                start = (
                    center[0] + math.cos(angle) * width * 0.32,
                    center[1] + math.sin(angle) * height * 0.32,
                )
                end = (
                    center[0] + math.cos(angle) * width * 0.55,
                    center[1] + math.sin(angle) * height * 0.55,
                )
                pygame.draw.line(overlay, (45, 35, 27, 150), start, end, 1)
        screen.blit(overlay, (0, 0))

    def _draw_explosions(self, screen):
        """绘制由真实健康值下降触发的小型爆炸，不改变实体终局。"""
        now = time.perf_counter()
        active = {}
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        for identifier, (position, started) in self._explosions.items():
            position = self._display_effect_positions.get(identifier, position)
            progress = (now - started) / self.EXPLOSION_DURATION_SECONDS
            if not 0.0 <= progress < 1.0:
                continue
            active[identifier] = (position, started)
            lon, lat, altitude = position
            if self.view_3d:
                x, y = self._geo_to_screen_3d_float(lon, lat, altitude)
            else:
                x, y = self._geo_to_screen_linear_float(lon, lat)
            center = (int(round(x)), int(round(y)))
            expansion = min(1.0, progress / 0.42)
            radius = max(4, int(round(5 + 17 * expansion)))
            alpha = max(0, int(230 * (1.0 - progress)))
            pygame.draw.circle(
                overlay, (255, 119, 28, min(210, alpha)), center, radius,
            )
            pygame.draw.circle(
                overlay, (255, 226, 105, min(255, alpha + 20)),
                center, max(2, int(radius * (0.52 - 0.22 * progress))),
            )
            if progress > 0.28:
                smoke_alpha = max(0, int(145 * (1.0 - progress)))
                for puff in range(3):
                    puff_angle = math.radians(identifier * 13 + puff * 118)
                    puff_center = (
                        int(round(center[0] + math.cos(puff_angle) * radius * 0.42)),
                        int(round(center[1] - radius * 0.28
                                  + math.sin(puff_angle) * radius * 0.20)),
                    )
                    pygame.draw.circle(
                        overlay, (48, 45, 42, smoke_alpha), puff_center,
                        max(3, int(radius * (0.25 + puff * 0.04))),
                    )
            ray_length = radius + 5
            for ray in range(0, 360, 60):
                angle = math.radians(ray + identifier % 31)
                inner = (
                    center[0] + math.cos(angle) * radius * 0.45,
                    center[1] + math.sin(angle) * radius * 0.45,
                )
                outer = (
                    center[0] + math.cos(angle) * ray_length,
                    center[1] + math.sin(angle) * ray_length,
                )
                pygame.draw.line(overlay, (255, 179, 62, alpha), inner, outer, 2)
        self._explosions = active
        screen.blit(overlay, (0, 0))

    def _draw_entities_3d(self, screen, entities):
        """按相机深度绘制实体、真实高度垂线和悬停信息。"""
        marker_drawers = {
            21000: self.draw_destroyer,
            21001: self.draw_assault,
            21002: self.draw_scavenger,
            9202: self.draw_eye,
            9400: self.draw_hive,
            9500: self.draw_dark_hive,
            9600: self.draw_sentry_hive,
            24000: self.draw_swarm,
        }
        drawable = []
        for data in entities.values():
            if data.get("type") == 35000:
                continue
            if data.get("type") == 9202 and not self.show_satellite:
                continue
            if not data.get("isVisible", False) or data.get("health", 0) <= 0:
                continue
            position = data.get("position", {})
            try:
                lon = float(position.get("lon", 0.0))
                lat = float(position.get("lat", 0.0))
                altitude = float(position.get("alt", 0.0))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (lon, lat, altitude)):
                continue
            altitude = max(0.0, altitude)
            entity_type = data.get("type")
            # 天眼是全局支援实体，在3D视图中固定显示于地图中心上空。
            if entity_type == 9202:
                lon = (self.SCENE_LON_MIN + self.SCENE_LON_MAX) / 2.0
                lat = (self.SCENE_LAT_MIN + self.SCENE_LAT_MAX) / 2.0
                altitude = max(altitude, 12000.0)
            ground = self._geo_to_screen_3d_float(lon, lat, 0.0)
            point = self._geo_to_screen_3d_float(lon, lat, altitude)
            drawer = marker_drawers.get(entity_type)
            drawable.append((self._camera_depth(lon, lat), data, lon, lat,
                             altitude, ground, point, drawer))

        # 探测范围和高度垂线先画；通信链路叠在范围圈之上、单位图标之下。
        for _, data, lon, lat, _, ground, point, _ in drawable:
            if data.get("type") == 9600:
                self._draw_3d_range_ring(screen, lon, lat, 20.0, (108, 155, 126))
            if data.get("side") != 0 and data.get("type") != 24000:
                stem_color = self.SIDE_COLORS.get(data.get("side"), self.NEUTRAL_COLOR)
                pygame.draw.line(screen, (*stem_color,), ground, point, 1)
                pygame.draw.circle(screen, self.GRID_COLOR, ground, 2)

        self._draw_communication_links(screen, entities)

        self._entity_screen_positions = {}
        altitude_labels = []
        for _, data, _, _, altitude, _, point, drawer in sorted(drawable, key=lambda item: item[0]):
            x, y = point
            if x < -60 or x > self.MAP_W + 60 or y < -60 or y > self.MAP_H + 60:
                continue
            entity_type = data.get("type")
            radius = self.LIVE_BACKGROUND_RADII.get(entity_type, self.draw_radius + 2)
            if entity_type == 9202:
                radius = self.EYE_BACKGROUND_RADIUS
            # 红方H/M/L依靠模型颜色区分，不再叠加每实体圆形阵营底圈。
            if data.get("side") != 0 and entity_type not in (9400, 9500, 9600, 24000):
                self._draw_side_background(screen, x, y, data.get("side"), radius)
            if entity_type == 9202:
                pygame.draw.circle(screen, self.EYE_EDGE_COLOR, (x, y),
                                   self.EYE_BACKGROUND_RADIUS + 4, 3)
            if drawer:
                size = self.LIVE_MARKER_SIZES.get(entity_type, 18)
                if entity_type == 9202:
                    size = self.EYE_MARKER_SIZE
                if entity_type in (21000, 21001, 21002):
                    self._draw_platform_model(screen, x, y, size, drawer, data)
                else:
                    drawer(screen, x, y, size)
            else:
                color = self.SIDE_COLORS.get(data.get("side"), self.NEUTRAL_COLOR)
                pygame.draw.circle(screen, color, (x, y), self.draw_radius)

            identifier = data.get("id")
            self._entity_screen_positions[identifier] = (x, y, data, altitude)
            if self.show_altitude_labels and data.get("type") in (21000, 21001, 21002, 24000):
                text = self.font.render(f"{altitude / 1000.0:.1f} km", True, self.TEXT_COLOR)
                altitude_labels.append((text, (x + 10, y - 12)))

        for text, position in altitude_labels:
            screen.blit(text, position)

        # 鼠标悬停提供实体类型、编号和精确高度，避免默认标签淹没高密度态势。
        nearest = None
        for identifier, (x, y, data, altitude) in self._entity_screen_positions.items():
            distance_sq = (x - self.mouse_x) ** 2 + (y - self.mouse_y) ** 2
            if distance_sq <= 28 ** 2 and (nearest is None or distance_sq < nearest[0]):
                nearest = (distance_sq, identifier, x, y, data, altitude)
        if nearest is not None:
            _, identifier, x, y, data, altitude = nearest
            pygame.draw.circle(screen, self.TEXT_COLOR, (x, y), 31, 2)
            label = self.TYPE_LABELS.get(data.get("type"), str(data.get("nameChn", "Entity")))
            try:
                stage = int(float(data.get("stage", -1)))
            except (TypeError, ValueError):
                stage = -1
            stage_label = self.STAGE_LABELS.get(stage, str(stage))
            lines = [
                f"{label}  ID {identifier}",
                f"ALT {altitude:.0f} m ({altitude / 1000.0:.2f} km)",
                f"STAGE {stage_label}",
            ]
            width = max(self.font.size(line)[0] for line in lines) + 16
            panel_height = len(lines) * 18 + 10
            panel = pygame.Surface((width, panel_height), pygame.SRCALPHA)
            panel.fill((*self.PANEL_COLOR, 235))
            pygame.draw.rect(panel, self.PANEL_BORDER_COLOR, panel.get_rect(), 1)
            for index, line in enumerate(lines):
                panel.blit(self.font.render(line, True, self.TEXT_COLOR), (8, 5 + 18 * index))
            px = min(max(4, x + 20), self.MAP_W - width - 4)
            py = min(max(4, y - panel_height - 4), self.MAP_H - panel_height - 4)
            screen.blit(panel, (px, py))

    def _draw_entities(self, screen, entities, *, show_all=False):
        """先绘制范围和所有底圆，再绘制所有图标，防止底圆遮挡邻近单位。"""
        if self.view_3d and not show_all:
            self._draw_entities_3d(screen, entities)
            return
        marker_drawers = {
            21000: self.draw_destroyer,
            21001: self.draw_assault,
            21002: self.draw_scavenger,
            9202: self.draw_eye,
            9400: self.draw_hive,
            9500: self.draw_dark_hive,
            9600: self.draw_sentry_hive,
            24000: self.draw_swarm,
        }
        visible_entities = []
        for data in entities.values():
            if data.get("type") == 35000:
                continue
            if data.get("type") == 9202 and not self.show_satellite:
                continue
            if not show_all and (not data.get("isVisible", False) or data.get("health", 0) <= 0):
                continue
            pos = data.get("position", {})
            lon, lat = pos.get("lon", 0), pos.get("lat", 0)
            entity_type = data.get("type")
            if entity_type == 9202 and not show_all:
                # 天眼作为全局支援标记，固定在坐标轴原点；只调整显示位置。
                x, y = self._world_to_screen(*self._coordinate_world_center())
            elif self.use_projection:
                x, y = self._geo_to_screen_projection(lon, lat)
            else:
                if show_all:
                    x, y = self._geo_to_screen_linear(lon, lat)
                else:
                    x, y = self._geo_to_screen_linear_float(lon, lat)
            if x < -50 or x > self.MAP_W + 50 or y < -50 or y > self.MAP_H + 50:
                continue
            drawer = marker_drawers.get(entity_type)
            visible_entities.append((data, x, y, drawer))

            # 哨卫巢视野范围属于地图辅助线，位于全部单位底圆和图标下方。
            if data.get("type") == 9600:
                self._draw_2d_range_ring(
                    screen, lon, lat, 20.0, (108, 155, 126),
                )

        if not show_all:
            self._draw_communication_links(screen, entities)

        for data, x, y, drawer in visible_entities:
            entity_type = data.get("type")
            if show_all:
                radius = self.MARKER_BACKGROUND_RADIUS if drawer else self.draw_radius + 4
                if entity_type == 9202:
                    radius = self.EYE_BACKGROUND_RADIUS
                elif entity_type == 24000:
                    radius = self.SWARM_BACKGROUND_RADIUS
            else:
                radius = self.LIVE_BACKGROUND_RADII.get(entity_type, self.draw_radius + 2)
            if show_all or (
                data.get("side") != 0 and entity_type not in (9400, 9500, 9600, 24000)
            ):
                self._draw_side_background(screen, x, y, data.get("side"), radius)

        # 天眼独立强调边位于图标下方，外沿比其阵营底圆再大一圈。
        for data, x, y, _ in visible_entities:
            if data.get("type") == 9202:
                pygame.draw.circle(screen, self.EYE_EDGE_COLOR, (x, y),
                                   self.EYE_BACKGROUND_RADIUS + 4, 3)

        labels = []
        for data, x, y, drawer in visible_entities:
            if drawer:
                entity_type = data.get("type")
                if show_all:
                    size = 50
                    if entity_type == 9202:
                        size = self.EYE_MARKER_SIZE
                    elif entity_type == 24000:
                        size = self.SWARM_MARKER_SIZE
                else:
                    size = self.LIVE_MARKER_SIZES.get(entity_type, 18)
                if not show_all and entity_type in (21000, 21001, 21002):
                    self._draw_platform_model(screen, x, y, size, drawer, data)
                else:
                    drawer(screen, x, y, size)
            else:
                color = self.SIDE_COLORS.get(data.get("side"), self.NEUTRAL_COLOR)
                pygame.draw.circle(screen, color, (x, y), self.draw_radius)
                name = data.get("nameChn", str(data.get("id", "")))
                text_surface = self.font.render(str(name), True, color)
                labels.append((text_surface, text_surface.get_rect(center=(x, y - 12))))
        for text_surface, text_rect in labels:
            screen.blit(text_surface, text_rect)

    def _configure_viewport(self, width, height):
        """按视口尺寸重建二维比例；三维投影仍使用固定场景边界。"""
        self.MAP_W = max(1, int(width))
        self.MAP_H = max(1, int(height))
        self.LON_MIN, self.LON_MAX = self.SCENE_LON_MIN, self.SCENE_LON_MAX
        self.LAT_MIN, self.LAT_MAX = self.SCENE_LAT_MIN, self.SCENE_LAT_MAX
        self._init_lonlat_pixel_ratio()
        self._recalculate_lonlat_boundary()

    def _load_scenic_background(self):
        """在实时X11进程中加载地景；离屏组织方预览不触碰外部素材。"""
        if not self.background_image_path.is_file():
            self._background_master = None
            return
        try:
            self._background_master = pygame.image.load(
                str(self.background_image_path),
            ).convert()
        except pygame.error:
            self._background_master = None

    def _paint_scenic_background(self, surface):
        """按cover方式铺设地景；同尺寸零旋转时保持1:1像素绘制。"""
        surface.fill(self.BACKGROUND_COLOR)
        image = self._background_master
        if image is None:
            return
        target_width, target_height = surface.get_size()
        source_width, source_height = image.get_size()
        # The caller owns the complete display transform.  Honour an explicit
        # background rotation in both 2D and 3D so a rotated overview keeps the
        # terrain, units, trajectories and effects in the same orientation.
        rotation = self.background_rotation_deg
        angle = math.radians(rotation)
        # 先把源图放大到足以覆盖旋转后的整个视口，避免四角露底色。
        required_width = (
            abs(math.cos(angle)) * target_width
            + abs(math.sin(angle)) * target_height
        )
        required_height = (
            abs(math.sin(angle)) * target_width
            + abs(math.cos(angle)) * target_height
            + 2.0 * abs(self.background_offset_y)
        )
        scale = max(
            required_width / max(1, source_width),
            required_height / max(1, source_height),
        )
        scaled_size = (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        )
        if scaled_size == (source_width, source_height):
            scaled = image.copy()
        else:
            scaled = pygame.transform.smoothscale(image, scaled_size)
        if abs(rotation) > 1e-6:
            scaled = pygame.transform.rotate(scaled, rotation)
            scaled_size = scaled.get_size()
        if self.background_opacity < 1.0:
            scaled.set_alpha(int(255 * self.background_opacity))
        surface.blit(
            scaled,
            ((target_width - scaled_size[0]) // 2,
             (target_height - scaled_size[1]) // 2
             + int(round(self.background_offset_y))),
        )
        haze = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        haze_override = getattr(self, "background_haze_alpha", None)
        haze_alpha = (
            int(haze_override)
            if haze_override is not None
            else (0 if self.geographic_layout == "fujian_taiwan" else (
                42 if self.view_3d else 96
            ))
        )
        if haze_alpha > 0:
            haze.fill((222, 234, 239, haze_alpha))
            surface.blit(haze, (0, 0))

    def _geographic_display_entities(self, entities):
        """Map M01 coordinates onto Fujian, the strait and Taiwan for display only."""
        if getattr(self, "geographic_layout", None) != "fujian_taiwan":
            return entities

        lon_span = max(1e-9, self.SCENE_LON_MAX - self.SCENE_LON_MIN)
        lat_span = max(1e-9, self.SCENE_LAT_MAX - self.SCENE_LAT_MIN)

        def taiwan_x(u, screen_y):
            relative = max(0.0, min(1.0, (u - 0.55) / 0.31))
            # Taiwan bends southwest toward its southern tip; keep a land margin.
            center = 0.86 - 0.16 * screen_y
            return center + (relative - 0.50) * 0.07

        red_types = (21000, 21001, 21002)
        for identifier, data in entities.items():
            if data.get("side") != 0 or data.get("type") not in red_types:
                continue
            position = data.get("position", {})
            try:
                initial = (
                    float(position.get("lon", 0.0)),
                    float(position.get("lat", 0.0)),
                    float(position.get("alt", 0.0)),
                )
            except (TypeError, ValueError):
                continue
            self._geographic_initial_positions.setdefault(int(identifier), initial)

        for entity_type in red_types:
            initial_lons = [
                position[0]
                for identifier, position in self._geographic_initial_positions.items()
                if entities.get(identifier, {}).get("type") == entity_type
            ]
            if initial_lons and entity_type not in self._geographic_initial_type_ranges:
                self._geographic_initial_type_ranges[entity_type] = (
                    min(initial_lons), max(initial_lons),
                )

        displayed = {}
        for identifier, data in entities.items():
            position = data.get("position", {})
            try:
                lon = float(position.get("lon", 0.0))
                lat = float(position.get("lat", 0.0))
                altitude = float(position.get("alt", 0.0))
            except (TypeError, ValueError):
                displayed[identifier] = data
                continue
            u = max(0.0, min(1.0, (lon - self.SCENE_LON_MIN) / lon_span))
            v = max(0.0, min(1.0, (lat - self.SCENE_LAT_MIN) / lat_span))
            screen_y = 0.86 - 0.72 * v
            fujian_edge_x = 0.18 + 0.18 * (1.0 - screen_y)
            if u <= 0.3364:
                screen_x = 0.08 + (fujian_edge_x - 0.08) * (u / 0.3364)
            elif u < 0.55:
                progress = (u - 0.3364) / (0.55 - 0.3364)
                screen_x = fujian_edge_x + (
                    taiwan_x(0.55, screen_y) - fujian_edge_x
                ) * progress
            else:
                screen_x = taiwan_x(u, screen_y)

            entity_type = data.get("type")
            initial = self._geographic_initial_positions.get(int(identifier))
            if data.get("side") == 0 and entity_type in red_types and initial:
                initial_u = max(
                    0.0,
                    min(1.0, (initial[0] - self.SCENE_LON_MIN) / lon_span),
                )
                if initial_u <= 0.3364:
                    type_min, type_max = self._geographic_initial_type_ranges.get(
                        entity_type, (initial[0], initial[0]),
                    )
                    if type_max - type_min > 1e-9:
                        within_type = (initial[0] - type_min) / (type_max - type_min)
                    else:
                        within_type = 0.5
                    # Depth is measured from the inland edge toward the coast.
                    # Keep small within-type spreads while maintaining H/M/L order.
                    depth_center = {21000: 0.18, 21001: 0.50, 21002: 0.82}[entity_type]
                    depth = depth_center + (within_type - 0.5) * 0.16
                    launch_x = 0.08 + (fujian_edge_x - 0.08) * depth

                    center_lat = (lat + initial[1]) * 0.5
                    east_km = (
                        (lon - initial[0]) * 111.32
                        * math.cos(math.radians(center_lat))
                    )
                    north_km = (lat - initial[1]) * 111.32
                    travelled_km = math.hypot(east_km, north_km)
                    # Hold the requested order on the pad, then smoothly hand
                    # control back to the real M01 trajectory over 2-22 km.
                    trajectory_mix = max(
                        0.0, min(1.0, (travelled_km - 2.0) / 20.0),
                    )
                    trajectory_mix = trajectory_mix * trajectory_mix * (
                        3.0 - 2.0 * trajectory_mix
                    )
                    screen_x = launch_x + (screen_x - launch_x) * trajectory_mix

            geo_lon, geo_lat = self._screen_to_geo_linear(
                screen_x * self.MAP_W,
                screen_y * self.MAP_H,
            )
            mapped = data.copy()
            mapped["position"] = dict(
                position, lon=geo_lon, lat=geo_lat, alt=altitude,
            )
            displayed[identifier] = mapped
            self._display_effect_positions[int(identifier)] = (
                geo_lon, geo_lat, altitude,
            )
        return displayed

    def _draw_live_view(self, surface, entities, red_area, *, show_coordinate,
                        title, show_legend=False, show_hud=False):
        """在一个独立视口绘制当前帧，供主视图和俯视图复用。"""
        entities = self._geographic_display_entities(entities)
        counts = {}
        for data in entities.values():
            if data.get("side") != 0 or data.get("type") not in (21000, 21001, 21002):
                continue
            if data.get("health", 0) <= 0:
                continue
            try:
                stage = int(float(data.get("stage", -1)))
            except (TypeError, ValueError):
                stage = -1
            counts[stage] = counts.get(stage, 0) + 1
        self._stage_counts = counts

        # 3/4阶段会有大量移动实体。底图若也逐步重算，会无谓占用同一CPU核。
        # 相机参数或窗口尺寸改变时key自然变化并重建，否则只做一次整层blit。
        # Deployment and side-area polygons are intentionally absent from the UI.
        area_key = ()
        static_key = (
            surface.get_size(), bool(show_coordinate), bool(self.view_3d),
            round(float(self.zoom), 4), round(float(self.offset_x), 2),
            round(float(self.offset_y), 2), round(float(self.camera_yaw_deg), 2),
            round(float(self.camera_pitch_deg), 2),
            round(float(self.altitude_pixels_per_km), 2),
            round(float(self.view_rotation_deg), 2), area_key,
            bool(self._background_master), round(self.background_opacity, 2),
            round(float(self.background_rotation_deg), 2),
            round(float(self.background_offset_y), 2),
        )
        static_layer = self._cache_get(self._static_layer_cache, static_key)
        if static_layer is None:
            static_layer = pygame.Surface(surface.get_size())
            self._paint_scenic_background(static_layer)
            if show_coordinate:
                if self.view_3d:
                    self._draw_3d_floor(static_layer)
                else:
                    self._draw_coordinate_system(static_layer)

            self._cache_put(
                self._static_layer_cache, static_key, static_layer,
                self._static_layer_cache_limit,
            )
        surface.blit(static_layer, (0, 0))

        self._draw_craters(surface)
        self._draw_entities(surface, entities)
        self._draw_explosions(surface)
        if show_legend:
            self.draw_legend2(surface)
        if show_hud:
            self._draw_view_hud(surface)

        title_surface = self.font_legend.render(title, True, self.TEXT_COLOR)
        title_panel = pygame.Surface(
            (title_surface.get_width() + 18, title_surface.get_height() + 8),
            pygame.SRCALPHA,
        )
        title_panel.fill((*self.PANEL_COLOR, 220))
        pygame.draw.rect(title_panel, self.PANEL_BORDER_COLOR, title_panel.get_rect(), 1)
        title_panel.blit(title_surface, (9, 4))
        surface.blit(title_panel, ((surface.get_width() - title_panel.get_width()) // 2, 8))

    def _render_loop(self):
        """渲染主循环（在独立线程中运行）"""
        pygame.init()
        desktop = pygame.display.Info()
        window_width = max(1600, desktop.current_w - 30) if desktop.current_w else 1600
        window_height = max(900, desktop.current_h - 80) if desktop.current_h else 900
        screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("M01 全屏俯视图 - 福建 / 台湾")
        clock = pygame.time.Clock()
        self._load_scenic_background()

        # 发布容器不保证安装fontconfig/fc-list；内置字体无需扫描系统字体，
        # 可确保VNC/X11窗口及时完成首帧映射。图例和3D HUD均使用ASCII文本。
        font_path = str(self.ui_font_path) if self.ui_font_path.is_file() else None
        self.font = pygame.font.Font(font_path, 14)
        self.font_legend = pygame.font.Font(font_path, 18)
        self.view_3d = False
        self.view_rotation_deg = 0.0
        self.geographic_layout = "fujian_taiwan"
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.background_opacity = 1.0

        # 右侧俯视图使用独立投影状态，避免影响主视图的相机和缩放。
        top_renderer = Renderer(
            self.SCENE_LON_MIN, self.SCENE_LON_MAX,
            self.SCENE_LAT_MIN, self.SCENE_LAT_MAX,
            fps=self.fps,
        )
        self.zoom = 1.35
        top_renderer.view_3d = False
        top_renderer.view_rotation_deg = 45.0
        top_renderer.show_satellite = False
        top_renderer.zoom = 0.82
        top_renderer._background_master = None
        top_renderer.font = self.font
        top_renderer.font_legend = self.font_legend

        divider_width = 4

        def configure_views(width, height):
            width = max(1000, int(width))
            height = max(700, int(height))
            main_width = width
            top_width = 0
            self._configure_viewport(main_width, height)
            self.zoom = 1.0
            self.offset_x = 0.0
            self.offset_y = 0.0
            return (
                pygame.Surface((main_width, height)),
                pygame.Surface((1, height)),
                main_width,
                top_width,
                width,
                height,
            )

        main_surface, top_surface, main_width, top_width, window_width, window_height = \
            configure_views(window_width, window_height)

        # 先提交一个空白首帧，使窗口管理器在主仿真开始前映射窗口。
        screen.fill(self.BACKGROUND_COLOR)
        pygame.display.flip()
        self._ready_event.set()

        # 坐标系开关
        show_coordinate = False

        while self.running:
            # ========== 事件处理 ==========
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                # ---- 滚轮缩放（以鼠标位置为中心） ----
                elif event.type == pygame.MOUSEWHEEL:
                    mouse_x, _ = pygame.mouse.get_pos()
                    if mouse_x < main_width:
                        if event.y > 0:
                            self._zoom_at_mouse(self.zoom_in)
                        else:
                            self._zoom_at_mouse(self.zoom_out)
                    else:
                        top_renderer.zoom_in() if event.y > 0 else top_renderer.zoom_out()

                # ---- 键盘事件 ----
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_view()
                    elif event.key == pygame.K_c:
                        show_coordinate = not show_coordinate
                    elif event.key == pygame.K_v:
                        self.view_3d = False
                    elif event.key == pygame.K_h:
                        self.show_altitude_labels = not self.show_altitude_labels
                    elif event.key == pygame.K_LEFT and self.view_3d:
                        self.rotate_camera(yaw_delta=-5.0)
                    elif event.key == pygame.K_RIGHT and self.view_3d:
                        self.rotate_camera(yaw_delta=5.0)
                    elif event.key == pygame.K_UP and self.view_3d:
                        self.rotate_camera(pitch_delta=3.0)
                    elif event.key == pygame.K_DOWN and self.view_3d:
                        self.rotate_camera(pitch_delta=-3.0)
                    elif event.key == pygame.K_LEFTBRACKET and self.view_3d:
                        self.change_altitude_scale(0.85)
                    elif event.key == pygame.K_RIGHTBRACKET and self.view_3d:
                        self.change_altitude_scale(1.18)
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self._zoom_at_mouse(self.zoom_in)
                    elif event.key == pygame.K_MINUS:
                        self._zoom_at_mouse(self.zoom_out)

                # ---- 鼠标拖拽平移 ----
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and event.pos[0] < main_width:
                        self.dragging = True
                        self.drag_start_x, self.drag_start_y = event.pos
                        self.drag_offset_x = self.offset_x
                        self.drag_offset_y = self.offset_y

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False

                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_x, self.mouse_y = event.pos
                    top_renderer.mouse_x = event.pos[0] - main_width - divider_width
                    top_renderer.mouse_y = event.pos[1]
                    if self.dragging:
                        dx = event.pos[0] - self.drag_start_x
                        dy = event.pos[1] - self.drag_start_y
                        self.offset_x = self.drag_offset_x + dx
                        self.offset_y = self.drag_offset_y + dy

                # ---- 窗口尺寸变化事件 ----
                elif event.type == pygame.VIDEORESIZE:
                    requested_width = max(1000, event.w)
                    requested_height = max(700, event.h)
                    screen = pygame.display.set_mode(
                        (requested_width, requested_height), pygame.RESIZABLE,
                    )
                    (main_surface, top_surface, main_width, top_width,
                     window_width, window_height) = configure_views(
                        requested_width, requested_height,
                    )

            # 新仿真步必须先原样绘制并确认；步间可另画不回写引擎的显示插值。
            with self._frame_condition:
                if self._submitted_frame <= self._rendered_frame:
                    frame_sequence = None
                    entities = self.entities_data.copy()
                    frame_step = self.current_step
                else:
                    frame_sequence = self._submitted_frame
                    entities = self.entities_data.copy()
                    frame_step = self._pending_step
            if frame_sequence is None:
                # 完整仿真在巡航/交战阶段可能1秒以上才产出下一步。等待期间
                # 仅对显示态按N-U-E速度补帧；精确仿真快照仍由同步分支确认。
                now = time.perf_counter()
                if now - self._last_motion_present < 1.0 / self.motion_fps:
                    clock.tick(self.fps)
                    continue
                entities, display_phase, moving = self._motion_extrapolation(
                    entities, now,
                )
                if not moving and not self._has_active_explosions(now):
                    clock.tick(self.fps)
                    continue
                is_exact_frame = False
            else:
                display_phase = 0.0
                is_exact_frame = True

            frame_started = time.perf_counter()
            self.current_step = frame_step
            self._display_phase = display_phase
            top_renderer._motion_directions = dict(self._motion_directions)
            top_renderer._smoothed_motion_directions = dict(
                self._smoothed_motion_directions,
            )
            top_renderer._explosions = dict(self._explosions)
            top_renderer._craters = dict(self._craters)
            top_renderer.current_step = frame_step
            top_renderer._display_phase = display_phase

            self._draw_live_view(
                main_surface, entities, self.red_area,
                show_coordinate=False,
                title="M01 TOP DOWN - FICTIONAL STRAIT",
                show_legend=True, show_hud=True,
            )

            screen.blit(main_surface, (0, 0))

            pygame.display.flip()
            presented_at = time.perf_counter()
            render_ms = (presented_at - frame_started) * 1000.0
            self._render_time_ms = (
                render_ms if self._render_time_ms <= 0.0
                else self._render_time_ms * 0.90 + render_ms * 0.10
            )
            if self._last_present_time is not None:
                interval = presented_at - self._last_present_time
                if interval > 0.0:
                    rate = 1.0 / interval
                    self._present_rate_hz = (
                        rate if self._present_rate_hz <= 0.0
                        else self._present_rate_hz * 0.90 + rate * 0.10
                    )
            self._last_present_time = presented_at
            self._last_motion_present = presented_at
            if is_exact_frame:
                with self._frame_condition:
                    self._rendered_frame = max(self._rendered_frame, frame_sequence)
                    self._frame_condition.notify_all()
            clock.tick(self.fps)

        with self._frame_condition:
            self.running = False
            self._frame_condition.notify_all()
        pygame.quit()

    def km_to_pixel_radius(self, lon: float, lat: float, range_km: float) -> float:
        """
        1、根据场景坐标与公里数的近似对应关系，计算目标距离对应的横纵坐标跨度
        2、根据当前渲染比例，分别计算横纵方向上的像素半径
        3、两个方向取平均
        """

        lat_rad = math.radians(lat)
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(lat_rad)

        range_lat = range_km / km_per_deg_lat
        range_lon = range_km / km_per_deg_lon

        r_lat = range_lat / self.lat_per_pixel
        r_lon = range_lon / self.lon_per_pixel
        return (r_lat + r_lon) / 2.0

    def draw_destroyer(self, surf, cx, cy, size=50):
        """H：带机腹阴影、高光弹体和大后掠翼的高性能导弹。"""
        half = size / 2.0
        color = self.TYPE_COLORS[21000]
        light = (229, 91, 116)
        dark = (105, 18, 43)
        shadow = (92, 105, 118, 120)
        offset = max(1, int(size * 0.06))
        body = [
            (cx, cy - half),
            (cx + size * 0.13, cy - size * 0.20),
            (cx + size * 0.12, cy + size * 0.39),
            (cx, cy + half),
            (cx - size * 0.12, cy + size * 0.39),
            (cx - size * 0.13, cy - size * 0.20),
        ]
        wings = [
            (cx - size * 0.10, cy - size * 0.02),
            (cx - size * 0.48, cy + size * 0.30),
            (cx - size * 0.11, cy + size * 0.20),
            (cx + size * 0.11, cy + size * 0.20),
            (cx + size * 0.48, cy + size * 0.30),
            (cx + size * 0.10, cy - size * 0.02),
        ]
        rear_fins = [
            (cx - size * 0.10, cy + size * 0.29),
            (cx - size * 0.29, cy + size * 0.47),
            (cx - size * 0.08, cy + size * 0.41),
            (cx + size * 0.08, cy + size * 0.41),
            (cx + size * 0.29, cy + size * 0.47),
            (cx + size * 0.10, cy + size * 0.29),
        ]

        def shifted(points):
            return [(x + offset, y + offset) for x, y in points]

        pygame.draw.polygon(surf, shadow, shifted(wings))
        pygame.draw.polygon(surf, shadow, shifted(rear_fins))
        pygame.draw.polygon(surf, shadow, shifted(body))
        pygame.draw.polygon(surf, dark, wings)
        pygame.draw.polygon(surf, color, rear_fins)
        pygame.draw.polygon(surf, color, body)
        pygame.draw.polygon(surf, dark, body, 1)
        # 右半侧阴影与左侧高光让圆柱弹体在小尺寸下仍有体积感。
        pygame.draw.polygon(surf, dark, [
            (cx, cy - half + size * 0.03),
            (cx + size * 0.13, cy - size * 0.20),
            (cx + size * 0.12, cy + size * 0.39),
            (cx, cy + half - size * 0.03),
        ])
        pygame.draw.line(surf, light,
                         (cx - size * 0.045, cy - size * 0.34),
                         (cx - size * 0.045, cy + size * 0.30),
                         max(1, int(size * 0.055)))
        pygame.draw.polygon(surf, (246, 167, 180), [
            (cx, cy - half),
            (cx - size * 0.07, cy - size * 0.26),
            (cx + size * 0.04, cy - size * 0.30),
        ])
        nozzle = pygame.Rect(int(cx - size * 0.08), int(cy + size * 0.39),
                             max(3, int(size * 0.16)), max(2, int(size * 0.08)))
        pygame.draw.ellipse(surf, (50, 56, 64), nozzle)

    def draw_assault(self, surf, cx, cy, size=50):
        """M：带圆柱明暗、短翼和尾喷口的低性能导弹。"""
        half = size / 2.0
        color = self.TYPE_COLORS[21001]
        light = (74, 177, 211)
        dark = (0, 57, 81)
        shadow = (88, 105, 117, 115)
        offset = max(1, int(size * 0.06))
        body = [
            (cx, cy - half),
            (cx + size * 0.14, cy - size * 0.17),
            (cx + size * 0.14, cy + size * 0.39),
            (cx + size * 0.08, cy + half),
            (cx - size * 0.08, cy + half),
            (cx - size * 0.14, cy + size * 0.39),
            (cx - size * 0.14, cy - size * 0.17),
        ]
        fins = [
            (cx - size * 0.12, cy + size * 0.12),
            (cx - size * 0.35, cy + size * 0.34),
            (cx - size * 0.12, cy + size * 0.27),
            (cx + size * 0.12, cy + size * 0.27),
            (cx + size * 0.35, cy + size * 0.34),
            (cx + size * 0.12, cy + size * 0.12),
        ]
        pygame.draw.polygon(surf, shadow,
                            [(x + offset, y + offset) for x, y in fins])
        pygame.draw.polygon(surf, shadow,
                            [(x + offset, y + offset) for x, y in body])
        pygame.draw.polygon(surf, dark, fins)
        pygame.draw.polygon(surf, color, body)
        pygame.draw.polygon(surf, dark, body, 1)
        pygame.draw.polygon(surf, dark, [
            (cx, cy - half + size * 0.03),
            (cx + size * 0.14, cy - size * 0.17),
            (cx + size * 0.14, cy + size * 0.39),
            (cx + size * 0.08, cy + half),
            (cx, cy + half),
        ])
        pygame.draw.line(surf, light,
                         (cx - size * 0.05, cy - size * 0.29),
                         (cx - size * 0.05, cy + size * 0.33),
                         max(1, int(size * 0.055)))
        pygame.draw.polygon(surf, (143, 220, 238), [
            (cx, cy - half),
            (cx - size * 0.075, cy - size * 0.24),
            (cx + size * 0.04, cy - size * 0.29),
        ])
        nozzle = pygame.Rect(int(cx - size * 0.08), int(cy + size * 0.42),
                             max(3, int(size * 0.16)), max(2, int(size * 0.08)))
        pygame.draw.ellipse(surf, (40, 52, 59), nozzle)

    def draw_scavenger(self, surf, cx, cy, size=50):
        """L：黄色固定翼飞翼无人机，便于和H/M快速区分。"""
        outline = self.TYPE_COLORS[21002]
        body_dark = (116, 78, 0)
        body_mid = (225, 158, 18)
        body_light = (255, 231, 122)
        shadow_offset = max(1, int(size * 0.055))

        silhouette = [
            (cx, cy - size * 0.50),
            (cx + size * 0.13, cy - size * 0.25),
            (cx + size * 0.49, cy - size * 0.08),
            (cx + size * 0.44, cy + size * 0.04),
            (cx + size * 0.17, cy + size * 0.10),
            (cx + size * 0.12, cy + size * 0.37),
            (cx + size * 0.05, cy + size * 0.49),
            (cx, cy + size * 0.40),
            (cx - size * 0.05, cy + size * 0.49),
            (cx - size * 0.12, cy + size * 0.37),
            (cx - size * 0.17, cy + size * 0.10),
            (cx - size * 0.44, cy + size * 0.04),
            (cx - size * 0.49, cy - size * 0.08),
            (cx - size * 0.13, cy - size * 0.25),
        ]
        shadow = [
            (x + shadow_offset, y + shadow_offset) for x, y in silhouette
        ]
        pygame.draw.polygon(surf, (18, 23, 28, 125), shadow)
        pygame.draw.polygon(surf, body_mid, silhouette)
        pygame.draw.polygon(
            surf, body_dark,
            [(cx, cy - size * 0.50),
             (cx + size * 0.13, cy - size * 0.25),
             (cx + size * 0.17, cy + size * 0.10),
             (cx, cy + size * 0.40)],
        )
        pygame.draw.polygon(surf, outline, silhouette,
                            max(1, int(size * 0.035)))

        # 上表面进气口、座舱/传感器窗及两片外倾尾翼。
        intake = [
            (cx - size * 0.12, cy - size * 0.16),
            (cx + size * 0.12, cy - size * 0.16),
            (cx + size * 0.08, cy - size * 0.04),
            (cx - size * 0.08, cy - size * 0.04),
        ]
        pygame.draw.polygon(surf, (12, 17, 22), intake)
        pygame.draw.line(
            surf, body_light,
            (cx - size * 0.18, cy - size * 0.22),
            (cx - size * 0.05, cy + size * 0.26),
            max(1, int(size * 0.025)),
        )
        for side in (-1, 1):
            fin = [
                (cx + side * size * 0.10, cy + size * 0.12),
                (cx + side * size * 0.24, cy + size * 0.30),
                (cx + side * size * 0.09, cy + size * 0.25),
            ]
            pygame.draw.polygon(surf, body_dark, fin)
            pygame.draw.line(
                surf, outline, fin[0], fin[1], max(1, int(size * 0.025)),
            )
        nozzle = pygame.Rect(
            int(cx - size * 0.07), int(cy + size * 0.35),
            max(2, int(size * 0.14)), max(2, int(size * 0.07)),
        )
        pygame.draw.ellipse(surf, (8, 12, 16), nozzle)

    def draw_eye(self, surf, cx, cy, size=50):
        """天眼：两段二次贝塞尔曲线 + 瞳孔"""
        k = size / 128.0

        def T(x, y): return (cx + (x - 64) * k, cy + (y - 64) * k)

        w = max(1, int(3 * k))
        # 上弧：从 (16,64) 经 (64,16) 到 (112,64)
        self.draw_bezier(surf, self.TYPE_COLORS[9202], T(16, 64), T(64, 16), T(112, 64), w)
        # 下弧：从 (112,64) 经 (64,112) 到 (16,64)
        self.draw_bezier(surf, self.TYPE_COLORS[9202], T(112, 64), T(64, 112), T(16, 64), w)

        pygame.draw.circle(surf, self.TYPE_COLORS[9202], T(64, 64), max(1, int(16 * k)))
        pygame.draw.circle(surf, (14, 22, 32), T(64, 64), max(1, int(7 * k)))

    def draw_hive(self, surf, cx, cy, size=50):
        """母巢：优先绘制透明写实贴图，缺失时回退程序化图标。"""
        if self._draw_hive_sprite(surf, cx, cy, size, 9400):
            return
        self._draw_hive_fallback(surf, cx, cy, size)

    def _draw_hive_fallback(self, surf, cx, cy, size=50):
        """母巢：贴地有机穹顶、根须和高光核心。"""
        center = (int(cx), int(cy))
        radius = max(5, int(size * 0.42))
        pygame.draw.ellipse(
            surf, (47, 59, 45),
            pygame.Rect(center[0] - radius, center[1] + radius // 3,
                        radius * 2, max(3, radius // 2)),
        )
        root_color = (91, 67, 25)
        for angle in range(0, 360, 60):
            radians = math.radians(angle)
            end = (
                int(cx + math.cos(radians) * radius * 1.22),
                int(cy + math.sin(radians) * radius * 0.72),
            )
            pygame.draw.line(surf, root_color, center, end, 2)
        pygame.draw.circle(surf, (91, 54, 13), center, radius)
        pygame.draw.circle(
            surf, self.TYPE_COLORS[9400],
            (center[0] - radius // 5, center[1] - radius // 5),
            max(3, int(radius * 0.72)),
        )
        pygame.draw.circle(
            surf, (230, 166, 67),
            (center[0] - radius // 3, center[1] - radius // 3),
            max(1, radius // 5),
        )
        pygame.draw.circle(surf, (58, 35, 8), center, radius, 1)

    def draw_sentry_hive(self, surf, cx, cy, size=50):
        """哨卫巢：优先绘制透明写实贴图，缺失时回退程序化图标。"""
        if self._draw_hive_sprite(surf, cx, cy, size, 9600):
            return
        self._draw_sentry_hive_fallback(surf, cx, cy, size)

    def _draw_sentry_hive_fallback(self, surf, cx, cy, size=50):
        """哨卫巢：六角地堡、中心塔和双侧感知臂。"""
        center = (int(cx), int(cy))
        radius = max(5, int(size * 0.43))
        pygame.draw.ellipse(
            surf, (42, 61, 49),
            pygame.Rect(center[0] - radius, center[1] + radius // 3,
                        radius * 2, max(3, radius // 2)),
        )
        points = [
            (int(cx + math.cos(math.radians(30 + index * 60)) * radius),
             int(cy + math.sin(math.radians(30 + index * 60)) * radius * 0.78))
            for index in range(6)
        ]
        pygame.draw.polygon(surf, (120, 149, 128), points)
        pygame.draw.polygon(surf, (16, 83, 48), points, 2)
        tower = pygame.Rect(
            center[0] - max(2, radius // 4), center[1] - radius,
            max(4, radius // 2), max(6, int(radius * 1.25)),
        )
        pygame.draw.rect(surf, (45, 116, 76), tower, border_radius=2)
        pygame.draw.line(
            surf, (201, 228, 210),
            (center[0] - radius, center[1] - radius // 2),
            (center[0] + radius, center[1] - radius // 2), 1,
        )
        pygame.draw.circle(surf, (156, 236, 184),
                           (center[0], center[1] - radius), max(1, radius // 5))

    def draw_dark_hive(self, surf, cx, cy, size=50):
        """暗巢：优先绘制透明写实贴图，缺失时回退程序化图标。"""
        if self._draw_hive_sprite(surf, cx, cy, size, 9500):
            return
        self._draw_dark_hive_fallback(surf, cx, cy, size)

    def _draw_dark_hive_fallback(self, surf, cx, cy, size=50):
        """暗巢：嵌入地表的暗色岩穴与发光内核。"""
        center = (int(cx), int(cy))
        radius = max(5, int(size * 0.45))
        pygame.draw.ellipse(
            surf, (38, 43, 39),
            pygame.Rect(center[0] - radius, center[1] + radius // 3,
                        radius * 2, max(3, radius // 2)),
        )
        factors = (1.0, 0.78, 1.08, 0.82, 1.0, 0.74, 1.05, 0.80)
        points = []
        for index, factor in enumerate(factors):
            angle = math.radians(-90 + index * 45)
            points.append((
                int(cx + math.cos(angle) * radius * factor),
                int(cy + math.sin(angle) * radius * factor * 0.82),
            ))
        pygame.draw.polygon(surf, (48, 27, 66), points)
        pygame.draw.polygon(surf, self.TYPE_COLORS[9500], points, 2)
        pygame.draw.circle(surf, (25, 13, 38), center, max(2, radius // 2))
        pygame.draw.circle(
            surf, (184, 113, 232),
            (center[0] - radius // 5, center[1] - radius // 5),
            max(1, radius // 5),
        )

    def draw_swarm(self, surf, cx, cy, size=50):
        """自爆虫：高对比亮洋红三点簇，深色描边适配复杂地景。"""
        radius = max(2, int(size * 0.18))
        offset = max(3, int(size * 0.24))
        outline = (75, 8, 30)
        highlight = (255, 151, 69)
        for dx, dy in ((0, -offset), (-offset, offset), (offset, offset)):
            center = (int(cx + dx), int(cy + dy))
            pygame.draw.circle(surf, outline, center, radius + 1)
            pygame.draw.circle(surf, self.TYPE_COLORS[24000], center, radius)
            pygame.draw.circle(
                surf, highlight,
                (center[0] - max(1, radius // 2),
                 center[1] - max(1, radius // 2)),
                1,
            )
        pygame.draw.circle(
            surf, (255, 218, 112), (int(cx), int(cy + offset // 2)),
            max(1, radius // 2),
        )

    def draw_legend2(self, surface,
                     margin=10,  # 距离窗口边缘的边距
                     icon_size=32,  # 图例中图标的大小
                     item_h=38,  # 每个条目的高度
                     panel_w=160,  # 图例面板宽度
                     bg_alpha=240,  # 背景透明度
                     *, labels=None, side_labels=None):

        if self.show_satellite:
            # 组织方静态预览保留原有完整图例和翻译键。
            icons = [
                ("毁灭者", self.draw_destroyer),
                ("突击者", self.draw_assault),
                ("清道夫", self.draw_scavenger),
                ("天眼", self.draw_eye),
                ("母巢", self.draw_hive),
                ("哨卫巢", self.draw_sentry_hive),
                ("暗巢", self.draw_dark_hive),
                ("自爆虫群", self.draw_swarm),
            ]
        else:
            # 实时UI使用紧凑英文图例，不显示天眼。
            icons = [
                ("H missile", self.draw_destroyer),
                ("M missile", self.draw_assault),
                ("L drone", self.draw_scavenger),
                ("Hive", self.draw_hive),
                ("Sentry", self.draw_sentry_hive),
                ("Hidden hive", self.draw_dark_hive),
                ("Interceptor", self.draw_swarm),
            ]
            icon_size = min(icon_size, 22)
            item_h = min(item_h, 28)
            panel_w = max(panel_w, 170)

        W, H = surface.get_size()
        n = len(icons)
        side_legend_h = 42
        panel_h = n * item_h + 12 + side_legend_h

        # 右上角坐标
        x0 = W - panel_w - margin
        y0 = margin

        # 半透明背景板
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((*self.PANEL_COLOR, bg_alpha))
        pygame.draw.rect(panel, (*self.PANEL_BORDER_COLOR, bg_alpha), panel.get_rect(), 1)
        surface.blit(panel, (x0, y0))

        # 逐个绘制图标 + 名称
        for i, (name, func) in enumerate(icons):
            # 图标中心
            cx = x0 + 10 + icon_size // 2
            cy = y0 + 8 + i * item_h + icon_size // 2
            func(surface, cx, cy, icon_size)
            if name == "天眼":
                pygame.draw.circle(surface, self.EYE_EDGE_COLOR, (cx, cy),
                                   icon_size // 2, 2)

            # 名称文字
            text = self.font_legend.render((labels or {}).get(name, name), True, self.TEXT_COLOR)
            tx = x0 + 10 + icon_size + 10
            ty = y0 + 8 + i * item_h + (icon_size - text.get_height()) // 2
            surface.blit(text, (tx, ty))

        # 底圆说明与单位类型图例分开，避免将图标自身颜色解释为阵营。
        divider_y = y0 + n * item_h + 8
        pygame.draw.line(surface, self.PANEL_BORDER_COLOR,
                         (x0 + 10, divider_y), (x0 + panel_w - 10, divider_y))
        for index, (side, label) in enumerate(((0, "Red"), (1, "Blue"))):
            cx = x0 + 16 + index * (panel_w // 2)
            cy = divider_y + 22
            self._draw_side_background(surface, cx, cy, side, 10)
            text = self.font_legend.render((side_labels or {}).get(label, label), True, self.TEXT_COLOR)
            surface.blit(text, (cx + 14, cy - text.get_height() // 2))

    def draw_single_legend(self,
                           bg_surface: pygame.Surface,
                           draw_shape_func: typing.Callable[[pygame.Surface, typing.Tuple[int, int]], pygame.Rect],
                           x:int,
                           y:int,
                           text: str,
                           text_left :int,
                           text_color: typing.Tuple[int, int, int])->int:
        shape_rect = draw_shape_func(bg_surface, (x, y))
        text_surface = self.font_legend.render(text, True, text_color)
        bg_surface.blit(text_surface, (text_left, y - text_surface.get_height()/2))
        return max(shape_rect.height, text_surface.get_height())

    def draw_single_legend2(self,
                           bg_surface: pygame.Surface,
                           draw_shape_func: typing.Callable[[pygame.Surface, int, int], pygame.Rect],
                           x:int,
                           y:int,
                           text: str,
                           text_left :int,
                           text_color: typing.Tuple[int, int, int])->int:
        shape_rect = draw_shape_func(bg_surface, x, y)
        text_surface = self.font_legend.render(text, True, text_color)
        bg_surface.blit(text_surface, (text_left, y - text_surface.get_height()/2))
        return max(shape_rect.height, text_surface.get_height())

    @staticmethod
    def draw_bezier(surface, color, p0, p1, p2, width=1, steps=30):
        """二次贝塞尔曲线，用线段逼近"""
        pts = []
        for i in range(steps + 1):
            t = i / steps
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
            pts.append((x, y))
        pygame.draw.lines(surface, color, False, pts, width)

    @staticmethod
    def get_poly_pts(cx, cy, r, n, rot_deg=-90):
        """生成正 n 边形顶点，中心 (cx,cy)，半径 r，起始角度 rot_deg"""
        pts = []
        for i in range(n):
            a = math.radians(rot_deg + i * 360 / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts

    @staticmethod
    def _draw_line(screen: pygame.Surface,
                   color: typing.Tuple[int, int, int],
                   center: typing.Tuple[int, int],
                   length: float,
                   rotation: float = 0,
                   width: int = 0)-> pygame.Rect:
        """
        绘制线段
        :param screen: 要绘制到哪里
        :param color: 颜色（填充或边线）
        :param center: 线段中心
        :param length: 线段长度
        :param rotation: 旋转角度，单位：弧度
        :param width: 边线粗细，width=0表示仅填充，width>0表示仅边线
        """

        x = int(length * math.cos(rotation) / 2)
        y = int(length * math.sin(rotation) / 2)
        return pygame.draw.line(screen, color, (center[0]+x, center[1]+y), (center[0]-x, center[1]-y), width)

    @staticmethod
    def draw_triangle(screen: pygame.Surface,
                      color: typing.Tuple[int, int, int],
                      center: typing.Tuple[int, int],
                      length_a: float,
                      length_b: float,
                      rotation:float=0,
                      width: int = 0)-> pygame.Rect:
        """
        绘制三角标记
        :param screen: 要绘制到哪里
        :param color: 颜色（填充或边线）
        :param center: 标记中心
        :param length_a: 标记长度
        :param length_b: 标记宽度
        :param rotation: 旋转角度，单位：弧度
        :param width: 边线粗细，width=0表示仅填充，width>0表示仅边线
        """

        local_points = [
            (length_a / 2, 0),  # 相对于中心
            (-length_a / 2, length_b / 2),
            (-length_a / 2, -length_b / 2),
        ]

        # 旋转并平移到中心
        points = []
        for dx, dy in local_points:
            # 旋转偏移量
            new_dx = dx * math.cos(rotation) - dy * math.sin(rotation)
            new_dy = dx * math.sin(rotation) + dy * math.cos(rotation)
            # 平移到中心
            points.append((center[0] + new_dx, center[1] + new_dy))

        return pygame.draw.polygon(screen, color, points, width)
