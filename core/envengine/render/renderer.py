# -*-coding:utf-8 -*-
import math
import os
import sys
import threading
import pygame
from pygame.draw import rect
from pyproj import Transformer
import typing


class Renderer:
    """
    实时渲染器 - 独立线程运行pygame窗口
    """

    def __init__(self, lon_min, lon_max, lat_min, lat_max, fps=60):
        self.entities_data = {}
        self.red_area = []
        self.red_areaHM = []
        self.running = False
        self.thread = None

        # 地图配置，基于不变形考虑，屏幕上每一个像素对应的经度和纬度必须维持一个固定的比例
        self.MIN_WINDOW_SIZE = 500
        self.LON_MIN, self.LON_MAX = lon_min, lon_max
        self.LAT_MIN, self.LAT_MAX = lat_min, lat_max
        self.lon_per_pixel = self.lat_per_pixel = 0 # 每像素对应的经度纬度（只在初始化时计算一次，后续保持不变）
        self.MAP_W, self.MAP_H = max(1000, self.MIN_WINDOW_SIZE), max(800, self.MIN_WINDOW_SIZE)
        # 只在初始化时计算一次每像素对应的经纬度比例，并记录地理中心
        self._init_lonlat_pixel_ratio()
        self._recalculate_lonlat_boundary()

        # ========== 缩放和平移参数 ==========
        self.zoom = 1.0
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

        # ========== 红蓝分割比例 ==========
        self.split_ratio = 0.5  # 红方占比，0.5表示50%

        # ========== 鼠标位置 ==========
        self.mouse_x = 0
        self.mouse_y = 0

        self.radar_angle = 0  # 雷达角度

        self.draw_radius = 8

    def start(self):
        """启动渲染线程"""
        self.running = True
        self.thread = threading.Thread(target=self._render_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止渲染线程"""
        self.running = False
        if self.thread:
            self.thread.join()

    def update_data(self, entities: dict):
        """
        更新态势数据（非阻塞，只保留最新）
        :param entities: {entity_id: {"position": {"lon": , "lat": , "alt": }}}
        """
        self.entities_data = entities

    def update_area(self, red_area: list[list[list[[float]]]], red_areaHM: list[list[list[[float]]]]):
        """
        更新区域数据（非阻塞，只保留最新）
        :param red_area: 红方区域
        """
        self.red_area = red_area[0] if red_area else []
        self.red_areaHM = red_areaHM[0] if red_areaHM else []

    def _init_lonlat_pixel_ratio(self):
        """只计算一次：每像素对应的经纬度（正方形像素，保持不变形）
        根据初始屏幕尺寸与场景经纬度范围确定比例，并记录场景地理中心，
        后续不再改变该比例。
        """
        lon_lat_pixel_ratio = 1  # 屏幕上一个像素对应的经度和纬度的比例，lon/lat

        self.lat_per_pixel = (self.LAT_MAX - self.LAT_MIN) / self.MAP_H
        self.lon_per_pixel = self.lat_per_pixel * lon_lat_pixel_ratio
        lon_pixel = (self.LON_MAX - self.LON_MIN) / self.lon_per_pixel
        if lon_pixel > self.MAP_W:
            self.lon_per_pixel = (self.LON_MAX - self.LON_MIN) / self.MAP_W
            self.lat_per_pixel = self.lon_per_pixel / lon_lat_pixel_ratio

        # 记录场景地理中心（恒定，不随窗口缩放变化）
        self.geo_center_lon = (self.LON_MAX + self.LON_MIN) / 2
        self.geo_center_lat = (self.LAT_MAX + self.LAT_MIN) / 2

    def _recalculate_lonlat_boundary(self):
        """根据屏幕尺寸重新计算对应的经纬度范围（仅重新居中，不改变每像素比例）
        使用初始化时确定的恒定 lon_per_pixel/lat_per_pixel 与地理中心，
        将场景地理中心对齐到当前窗口中心，保证窗口缩放时坐标轴（位于地理中心）
        始终居中、不发生偏移。
        """
        self.LON_MIN = self.geo_center_lon - self.lon_per_pixel * self.MAP_W / 2
        self.LON_MAX = self.geo_center_lon + self.lon_per_pixel * self.MAP_W / 2
        self.LAT_MIN = self.geo_center_lat - self.lat_per_pixel * self.MAP_H / 2
        self.LAT_MAX = self.geo_center_lat + self.lat_per_pixel * self.MAP_H / 2

    # ========== 缩放控制方法 ==========
    def zoom_in(self):
        """放大"""
        self.zoom = min(self.zoom + self.zoom_step, self.max_zoom)

    def zoom_out(self):
        """缩小"""
        self.zoom = max(self.zoom - self.zoom_step, self.min_zoom)

    def reset_view(self):
        """重置视图"""
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

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
        # 先将经纬度映射到世界坐标 (0~MAP_W, 0~MAP_H)
        wx = (lon - self.LON_MIN) / self.lon_per_pixel # 距离屏幕左边缘的像素数
        wy = (self.LAT_MAX - lat) / self.lat_per_pixel # 距离屏幕上边缘的像素数

        # 世界坐标转屏幕坐标（原点在屏幕中心）
        cx, cy = self.MAP_W // 2, self.MAP_H // 2
        sx = (wx - cx) * self.zoom + cx + self.offset_x
        sy = (wy - cy) * self.zoom + cy + self.offset_y
        return int(sx), int(sy)

    def _geo_to_screen_projection(self, lon, lat):
        """
        投影转换（带缩放和偏移）
        返回屏幕坐标，原点在屏幕中心
        """
        x, y = self.transformer.transform(lon, lat)
        wx = (x - self.xmin) / (self.xmax - self.xmin) * self.MAP_W
        wy = (self.ymax - y) / (self.ymax - self.ymin) * self.MAP_H

        cx, cy = self.MAP_W // 2, self.MAP_H // 2
        sx = (wx - cx) * self.zoom + cx + self.offset_x
        sy = (wy - cy) * self.zoom + cy + self.offset_y
        return int(sx), int(sy)

    def _world_to_screen(self, world_x, world_y):
        """
        将世界坐标（0~MAP_W, 0~MAP_H）转换为屏幕坐标（带缩放和偏移）
        原点在屏幕中心
        """
        cx, cy = self.MAP_W // 2, self.MAP_H // 2
        sx = (world_x - cx) * self.zoom + cx + self.offset_x
        sy = (world_y - cy) * self.zoom + cy + self.offset_y
        return int(sx), int(sy)

    def _screen_to_world(self, screen_x, screen_y):
        """
        将屏幕坐标转换为世界坐标（0~MAP_W, 0~MAP_H）
        """
        cx, cy = self.MAP_W // 2, self.MAP_H // 2
        wx = (screen_x - cx - self.offset_x) / self.zoom + cx if self.zoom != 0 else 0
        wy = (screen_y - cy - self.offset_y) / self.zoom + cy if self.zoom != 0 else 0
        return wx, wy

    def _screen_to_geo_linear(self, screen_x, screen_y):
        """
        将屏幕坐标转换为经纬度（线性映射）
        """
        wx, wy = self._screen_to_world(screen_x, screen_y)
        lon = wx * self.lon_per_pixel + self.LON_MIN
        lat = self.LAT_MAX - wy * self.lat_per_pixel
        return lon, lat

    def _draw_coordinate_system(self, screen):
        """
        绘制无限延伸的坐标系（与背景绑定）
        """
        cx, cy = self.MAP_W // 2, self.MAP_H // 2

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
        grid_color = (60, 60, 80)

        x = start_wx
        while x <= max_wx:
            sx, sy = self._world_to_screen(x, 0)
            if -100 <= sx <= self.MAP_W + 100:
                pygame.draw.line(screen, grid_color, (sx, 0), (sx, self.MAP_H), 1)
            x += grid_spacing_world

        y = start_wy
        while y <= max_wy:
            sx, sy = self._world_to_screen(0, y)
            if -100 <= sy <= self.MAP_H + 100:
                pygame.draw.line(screen, grid_color, (0, sy), (self.MAP_W, sy), 1)
            y += grid_spacing_world

        # ========== 绘制坐标轴（无限延伸） ==========
        # X轴（红色）- 水平方向
        axis_color = (255, 80, 80)
        p1 = self._world_to_screen(min_wx, cy)
        p2 = self._world_to_screen(max_wx, cy)
        pygame.draw.line(screen, axis_color, p1, p2, 2)

        # X轴箭头（右端）
        arrow_size = 10
        arrow_x = self.MAP_W - 20
        arrow_y = self._world_to_screen(0, cy)[1]
        if arrow_y < 0 or arrow_y > self.MAP_H:
            arrow_y = max(20, min(self.MAP_H - 20, arrow_y))
        # 绘制箭头
        pygame.draw.polygon(screen, axis_color, [
            (self.MAP_W - 10, arrow_y - arrow_size // 2),
            (self.MAP_W, arrow_y),
            (self.MAP_W - 10, arrow_y + arrow_size // 2)
        ])

        # Y轴（蓝色）- 垂直方向
        axis_color = (80, 80, 255)
        p1 = self._world_to_screen(cx, min_wy)
        p2 = self._world_to_screen(cx, max_wy)
        pygame.draw.line(screen, axis_color, p1, p2, 2)

        # Y轴箭头（上端）
        arrow_x = self._world_to_screen(cx, 0)[0]
        arrow_y = 20
        if arrow_x < 0 or arrow_x > self.MAP_W:
            arrow_x = max(20, min(self.MAP_W - 20, arrow_x))
        # 绘制箭头
        pygame.draw.polygon(screen, axis_color, [
            (arrow_x - arrow_size // 2, 10),
            (arrow_x, 0),
            (arrow_x + arrow_size // 2, 10)
        ])

        # ========== 绘制原点标记 ==========
        origin_screen = self._world_to_screen(cx, cy)
        if -20 <= origin_screen[0] <= self.MAP_W + 20 and -20 <= origin_screen[1] <= self.MAP_H + 20:
            pygame.draw.circle(screen, (255, 255, 0), origin_screen, 8, 2)
            pygame.draw.circle(screen, (255, 255, 0), origin_screen, 3)
            origin_text = self.font.render(f"O", True, (255, 255, 0))
            screen.blit(origin_text, (origin_screen[0] + 12, origin_screen[1] - 8))

        # ========== 显示X轴和Y轴标签（跟随坐标系） ==========
        x_label = self.font.render("X", True, (255, 80, 80))
        y_label = self.font.render("Y", True, (80, 80, 255))

        # X轴标签放在X轴箭头右侧（跟随箭头位置）
        # 箭头在屏幕右边缘，标签放在箭头右侧
        arrow_y = self._world_to_screen(0, cy)[1]
        if arrow_y < 0 or arrow_y > self.MAP_H:
            arrow_y = max(20, min(self.MAP_H - 20, arrow_y))
        # 标签放在箭头右侧
        screen.blit(x_label, (self.MAP_W - 20, arrow_y - 10))

        # Y轴标签放在Y轴箭头上方（跟随箭头位置）
        # 箭头在屏幕上边缘，标签放在箭头上方
        arrow_x = self._world_to_screen(cx, 0)[0]
        if arrow_x < 0 or arrow_x > self.MAP_W:
            arrow_x = max(20, min(self.MAP_W - 20, arrow_x))
        # 标签放在箭头上方
        screen.blit(y_label, (arrow_x - 10, 5))

        # ========== 绘制轴上的刻度标记（无限延伸） ==========
        # X轴刻度
        step = 100 / self.zoom if self.zoom != 0 else 100
        x_start = (min_wx // step) * step
        x_end = (max_wx // step) * step

        x = x_start
        while x <= x_end:
            sx, sy = self._world_to_screen(x, cy)
            if -20 <= sx <= self.MAP_W + 20 and -20 <= sy <= self.MAP_H + 20:
                pygame.draw.line(screen, (255, 255, 255),
                                 (sx, sy - 5), (sx, sy + 5), 1)
                if abs(x - cx) > 5:
                    offset = int(x - cx)
                    text = self.font.render(f"{offset}", True, (200, 200, 200))
                    screen.blit(text, (sx - 10, sy + 8))
            x += step

        # Y轴刻度
        y_start = (min_wy // step) * step
        y_end = (max_wy // step) * step

        y = y_start
        while y <= y_end:
            sx, sy = self._world_to_screen(cx, y)
            if -20 <= sx <= self.MAP_W + 20 and -20 <= sy <= self.MAP_H + 20:
                pygame.draw.line(screen, (255, 255, 255),
                                 (sx - 5, sy), (sx + 5, sy), 1)
                if abs(y - cy) > 5:
                    offset = int(cy - y)
                    text = self.font.render(f"{offset}", True, (200, 200, 200))
                    screen.blit(text, (sx + 8, sy - 6))
            y += step

    def _render_loop(self):
        """渲染主循环（在独立线程中运行）"""
        pygame.init()
        screen = pygame.display.set_mode((self.MAP_W, self.MAP_H), pygame.RESIZABLE)
        pygame.display.set_caption("仿真态势显示")
        clock = pygame.time.Clock()

        if sys.platform == 'win32':
            font_name = "simhei"
        else:
            font_name = "Noto Sans CJK SC"
        self.font = pygame.font.SysFont(font_name, 10)
        self.font_small = pygame.font.SysFont(font_name, 9)
        self.font_legend = pygame.font.SysFont(font_name, 20)

        # 坐标系开关
        show_coordinate = True

        while self.running:
            # ========== 事件处理 ==========
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                # ---- 滚轮缩放（以鼠标位置为中心） ----
                elif event.type == pygame.MOUSEWHEEL:
                    if event.y > 0:
                        self._zoom_at_mouse(self.zoom_in)
                    else:
                        self._zoom_at_mouse(self.zoom_out)

                # ---- 键盘事件 ----
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_view()
                    elif event.key == pygame.K_c:
                        show_coordinate = not show_coordinate
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self._zoom_at_mouse(self.zoom_in)
                    elif event.key == pygame.K_MINUS:
                        self._zoom_at_mouse(self.zoom_out)

                # ---- 鼠标拖拽平移 ----
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.dragging = True
                        self.drag_start_x, self.drag_start_y = event.pos
                        self.drag_offset_x = self.offset_x
                        self.drag_offset_y = self.offset_y

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False

                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_x, self.mouse_y = event.pos
                    if self.dragging:
                        dx = event.pos[0] - self.drag_start_x
                        dy = event.pos[1] - self.drag_start_y
                        self.offset_x = self.drag_offset_x + dx
                        self.offset_y = self.drag_offset_y + dy

                # ---- 窗口尺寸变化事件 ----
                elif event.type == pygame.VIDEORESIZE:
                    self.MAP_W, self.MAP_H = event.size

                    # 如果小于最小值，则强制修正
                    is_size_valid = True
                    if self.MAP_W < self.MIN_WINDOW_SIZE:
                        self.MAP_W = self.MIN_WINDOW_SIZE
                        is_size_valid = False
                    if self.MAP_H < self.MIN_WINDOW_SIZE:
                        self.MAP_H = self.MIN_WINDOW_SIZE
                        is_size_valid = False
                    if not is_size_valid:
                        # 重新设置窗口模式，应用限制后的尺寸
                        screen = pygame.display.set_mode((self.MAP_W, self.MAP_H), pygame.RESIZABLE)

                    # 重新计算经纬度边界，使地理中心重新对齐到新的窗口中心，
                    # 避免窗口缩放时坐标轴（位于地理中心）相对屏幕中心发生偏移
                    self._recalculate_lonlat_boundary()

            # 获取最新态势
            entities = self.entities_data.copy()

            # ========== 计算分割线的世界坐标位置 ==========
            split_line_x_world = self.MAP_W * self.split_ratio
            cx, cy = self.MAP_W // 2, self.MAP_H // 2
            split_screen_x = (split_line_x_world - cx) * self.zoom + cx + self.offset_x

            # ========== 绘制无限延伸的红蓝背景 ==========
            screen.fill((120, 40, 40))
            if split_screen_x < self.MAP_W:
                pygame.draw.rect(screen, (30, 40, 80),
                                 (max(0, split_screen_x), 0,
                                  self.MAP_W - max(0, split_screen_x), self.MAP_H))

            # 绘制分割线
            # if self.split_ratio > 0 and self.split_ratio < 1:
            #     pygame.draw.line(screen, (255, 255, 255),
            #                      (split_screen_x, 0), (split_screen_x, self.MAP_H), 0)

            # ========== 绘制坐标系（无限延伸，与背景绑定） ==========
            if show_coordinate:
                self._draw_coordinate_system(screen)

            # ========== 绘制红方区域多边形 ==========
            if len(self.red_area) >= 3:
                red_area_pixel = [list(self._geo_to_screen_linear(lon, lat)) for lon, lat in self.red_area]
                # pygame.draw.polygon(screen, (200, 50, 50), red_area_pixel)
                pygame.draw.polygon(screen, (255, 0, 0), red_area_pixel, 3)
            if len(self.red_areaHM) >= 3:
                red_area_pixel = [list(self._geo_to_screen_linear(lon, lat)) for lon, lat in self.red_areaHM]
                pygame.draw.polygon(screen, (255, 0, 0), red_area_pixel, 3)

            # ========== 绘制实体 ==========
            for entity_id, data in entities.items():
                if not data.get("isVisible", False):
                    continue

                if data.get("health", 0) <= 0:
                    continue

                pos = data.get('position', {})
                lon = pos.get('lon', 0)
                lat = pos.get('lat', 0)
                entity_type:int = data.get('type', 0)

                if self.use_projection:
                    x, y = self._geo_to_screen_projection(lon, lat)
                else:
                    x, y = self._geo_to_screen_linear(lon, lat)

                if x < -50 or x > self.MAP_W + 50 or y < -50 or y > self.MAP_H + 50:
                    continue

                color = (255, 120, 50) if data.get('side') == 0 else (50, 220, 255)
                name = data.get('nameChn', str(data.get('id', '')))
                match entity_type:
                    case 21000:
                        # 高性能飞行器
                        self.draw_missile_h(screen, (x,y))
                    case 21001:
                        # 低性能飞行器
                        self.draw_missile_m(screen, (x,y))
                    case 21002:
                        # 无人机
                        self.draw_missile_l(screen, (x,y))
                    case 9202:
                        # 卫星
                        self.draw_satellite(screen, (x,y))
                    case 9400:
                        # 目标
                        self.draw_target(screen, (x,y))
                    case 9500:
                        # 无人船
                        self.draw_autonomous_ship(screen, (x, y))
                    case 9600:
                        # 拦截阵地
                        self.draw_defensive(screen, (x,y))

                        # 雷达：按 200km 长度绘制探测范围圆形
                        pixel_radius = self.km_to_pixel_radius(lon, lat, 200)
                        pygame.draw.circle(screen, (0, 255, 0), (x, y), pixel_radius * self.zoom, 1)
                    case 24000:
                        # 拦截弹
                        self.draw_intercept(screen, (x,y))
                    case _:
                        pygame.draw.circle(screen, color, (x, y), self.draw_radius)
                        text_surface = self.font.render(str(name), True, color)
                        text_rect = text_surface.get_rect(center=(x, y - 12))
                        screen.blit(text_surface, text_rect)

                # 绘制名称
                # text_surface = self.font.render(str(name), True, color)
                # text_rect = text_surface.get_rect(center=(x, y - 12))
                # screen.blit(text_surface, text_rect)

            # ========== 右上角图例 ==========
            self.draw_legend(screen)

            # ========== 左下角显示鼠标位置坐标（以原点为中心） ==========
            # 获取鼠标位置对应的经纬度和世界坐标
            mouse_wx, mouse_wy = self._screen_to_world(self.mouse_x, self.mouse_y)
            mouse_lon, mouse_lat = self._screen_to_geo_linear(self.mouse_x, self.mouse_y)

            # 计算相对于原点（屏幕中心）的偏移量
            cx, cy = self.MAP_W // 2, self.MAP_H // 2

            # 世界坐标相对于原点的偏移
            world_offset_x = mouse_wx - cx
            world_offset_y = cy - mouse_wy  # Y轴向上为正

            # 构建坐标信息字符串
            coord_info = [
                f"鼠标位置 (原点为屏幕中心):",
                f"  世界坐标: ({world_offset_x:+.0f}, {world_offset_y:+.0f})",
                f"  经纬度: ({mouse_lon:.4f}, {mouse_lat:.4f})"
            ]

            # 绘制半透明背景（左下角）
            text_height = 16
            bg_width = 160
            bg_height = len(coord_info) * text_height + 10
            bg_x = 10
            bg_y = self.MAP_H - bg_height - 10

            # 半透明背景
            bg_surface = pygame.Surface((bg_width, bg_height))
            bg_surface.set_alpha(180)
            bg_surface.fill((0, 0, 0))
            screen.blit(bg_surface, (bg_x, bg_y))

            # 绘制文字
            for i, text in enumerate(coord_info):
                color = (200, 200, 200)
                if "鼠标位置" in text:
                    color = (255, 255, 100)
                text_surface = self.font_small.render(text, True, color)
                screen.blit(text_surface, (bg_x + 10, bg_y + 10 + i * text_height))

            # ========== UI信息（固定在屏幕右下角） ==========
            # 定义右下角信息区域
            line_height = 20

            # 计算实体数量
            alive_entities = {k: v for k, v in entities.items() if v.get('health', 0) > 0}
            entity_count = len(alive_entities)

            # 构建信息列表
            info_lines = [
                (f"缩放: {self.zoom:.1f}x", (255, 255, 255)),
                (f"坐标系: {'ON' if show_coordinate else 'OFF'}", (200, 200, 200)),
                (f"实体数: {entity_count}", (200, 200, 200)),
                ("操作: 滚轮缩放 | 拖拽平移", (180, 180, 180)),
                ("快捷键: 英文R重置 | 英文C开关坐标系", (180, 180, 180))
            ]

            # 计算背景大小
            max_width = 0
            for line, _ in info_lines:
                text_surface = self.font.render(line, True, (255, 255, 255))
                max_width = max(max_width, text_surface.get_width())

            bg_width = max_width + 20
            bg_height = len(info_lines) * line_height + 10

            # 绘制半透明背景（右下角）
            bg_surface = pygame.Surface((bg_width, bg_height))
            bg_surface.set_alpha(200)
            bg_surface.fill((0, 0, 0))
            screen.blit(bg_surface, (self.MAP_W - bg_width - 10, self.MAP_H - bg_height - 10))

            # 绘制信息文本
            for i, (line, color) in enumerate(info_lines):
                text_surface = self.font.render(line, True, color)
                screen.blit(text_surface, (self.MAP_W - bg_width, self.MAP_H - bg_height + i * line_height))

            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()

    def km_to_pixel_radius(self, lon: float, lat: float, range_km: float) -> float:
        """
        1、根据经纬度与公里数的近似对应关系，计算出目标距离对应的经度和纬度
        2、根据当前render的经纬度与对应的像素的关系，分别计算在经纬度方向上的像素
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

    def draw_missile_h(self,
                      screen: pygame.Surface,
                      center: typing.Tuple[int, int])->pygame.Rect:
        """绘制高性能弹"""
        return self.draw_triangle(screen, (255, 0, 0), center, self.draw_radius * 7, self.draw_radius * 2)

    def draw_missile_m(self,
                      screen: pygame.Surface,
                      center: typing.Tuple[int, int])->pygame.Rect:
        """绘制中性能弹"""
        return self.draw_triangle(screen, (232, 14, 217), center, self.draw_radius * 5, self.draw_radius * 2)

    def draw_missile_l(self,
                       screen: pygame.Surface,
                       center: typing.Tuple[int, int])->pygame.Rect:
        """绘制低性能弹"""
        return self.draw_triangle(screen, (79, 232, 14), center, self.draw_radius * 2, self.draw_radius * 2)

    def draw_satellite(self,
                       screen: pygame.Surface,
                       center: typing.Tuple[int, int])->pygame.Rect:
        """绘制卫星"""
        x = center[0]
        y = center[1]
        rec_length = self.draw_radius * 5
        first = pygame.draw.circle(screen, (255, 120, 50), center, self.draw_radius * 1.5)
        second = pygame.draw.rect(screen, (255, 120, 50), pygame.Rect(x - rec_length / 2, y - self.draw_radius * 0.5, rec_length, self.draw_radius))
        left = min(first.left, second.left)
        top = min(first.top, second.top)
        right = max(first.right, second.right)
        bottom = max(first.bottom, second.bottom)
        return pygame.Rect(left, top, right-left, bottom-top)

    def draw_target(self,
                       screen: pygame.Surface,
                       center: typing.Tuple[int, int])-> pygame.Rect:
        """绘制目标"""
        first = self._draw_line(screen, (50, 220, 255), center, self.draw_radius * 4, math.pi / 4,5)
        second = self._draw_line(screen, (50, 220, 255), center, self.draw_radius * 4, -math.pi / 4, 5)
        left = min(first.left, second.left)
        top = min(first.top, second.top)
        right = max(first.right, second.right)
        bottom = max(first.bottom, second.bottom)
        return pygame.Rect(left, top, right-left, bottom-top)

    def draw_defensive(self,
                       screen: pygame.Surface,
                       center: typing.Tuple[int, int])-> pygame.Rect:
        """绘制防御阵地"""
        x = center[0]
        y = center[1]
        return pygame.draw.rect(screen, (50, 220, 255), pygame.Rect(x - self.draw_radius, y - self.draw_radius, self.draw_radius * 2, self.draw_radius * 2))

    def draw_autonomous_ship(self,
                       screen: pygame.Surface,
                       center: typing.Tuple[int, int])-> pygame.Rect:
        """绘制无人船"""
        return pygame.draw.circle(screen, (50, 220, 255), center, self.draw_radius)  # 拦截范围

    def draw_intercept(self,
                       screen: pygame.Surface,
                       center: typing.Tuple[int, int])-> pygame.Rect:
        """绘制拦截弹"""
        return self.draw_triangle(screen, (50, 220, 255), center, self.draw_radius, self.draw_radius, math.pi)

    def draw_legend(self, screen: pygame.Surface):
        """绘制图例
        1、创建一个图例组件
        2、将组件放置到屏幕右上角
        """

        bg_width = 250
        bg_height = 250

        shape_center_x = 50
        shape_center_y = 20
        text_color = (255, 255, 255)
        text_left = shape_center_x + 50
        line_space = 20

        # 绘制半透明背景（右下角）
        bg_surface = pygame.Surface((bg_width, bg_height))
        bg_surface.set_alpha(200)
        bg_surface.fill((0, 0, 0))

        line_height = self.draw_single_legend(bg_surface, self.draw_missile_h, shape_center_x, shape_center_y, "高性能飞行器", text_left, text_color)
        shape_center_y += line_height / 2 + line_space

        line_height = self.draw_single_legend(bg_surface, self.draw_missile_m, shape_center_x, shape_center_y, "低性能飞行器", text_left, text_color)
        shape_center_y += line_height / 2 + line_space

        line_height = self.draw_single_legend(bg_surface, self.draw_missile_l, shape_center_x, shape_center_y, "无人机", text_left, text_color)
        shape_center_y += line_height / 2 + line_space

        line_height = self.draw_single_legend(bg_surface, self.draw_satellite, shape_center_x, shape_center_y, "卫星", text_left, text_color)
        shape_center_y += line_height / 2 + line_space

        line_height = self.draw_single_legend(bg_surface, self.draw_target, shape_center_x, shape_center_y, "目标", text_left, text_color)
        shape_center_y += line_height / 2 + line_space

        line_height = self.draw_single_legend(bg_surface, self.draw_defensive, shape_center_x, shape_center_y, "防御阵地", text_left, text_color)
        shape_center_y += line_height / 2 + line_space

        line_height = self.draw_single_legend(bg_surface, self.draw_autonomous_ship, shape_center_x, shape_center_y, "无人船", text_left, text_color)
        shape_center_y += line_height / 2 + line_space

        line_height = self.draw_single_legend(bg_surface, self.draw_intercept, shape_center_x, shape_center_y, "拦截弹", text_left, text_color)
        shape_center_y += line_height / 2 + line_space

        screen.blit(bg_surface, (self.MAP_W - bg_width - 10, 10))

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

    def _draw_line(self,
                   screen: pygame.Surface,
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

    def draw_regular_polygon(self,
                      screen: pygame.Surface,
                      color: typing.Tuple[int, int, int],
                      center: typing.Tuple[int, int],
                      sides: int,
                      radius: float,
                      rotation: float = 0,
                      width: int = 0)-> pygame.Rect:
        """
        绘制六边形
        :param screen: 要绘制到哪里
        :param color: 颜色（填充或边线）
        :param center: 多边形中心
        :param sides: 边数
        :param radius: 多边形半径，即center到多边形边点的距离
        :param rotation: 旋转角度，单位：弧度
        :param width: 边线粗细，width=0表示仅填充，width>0表示仅边线
        """

        if sides < 3:
            raise ValueError("边数必须 >= 3")

        points = []
        for i in range(sides):
            # 每个顶点间隔 2*pi/sides
            angle = rotation + i * 2 * math.pi / sides
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((int(x), int(y)))

        return pygame.draw.polygon(screen, color, points, width)

    def draw_triangle(self,
                      screen: pygame.Surface,
                      color: typing.Tuple[int, int, int],
                      center: typing.Tuple[int, int],
                      length_a: float,
                      length_b: float,
                      rotation:float=0,
                      width: int = 0)-> pygame.Rect:
        """
        绘制导弹
        :param screen: 要绘制到哪里
        :param color: 颜色（填充或边线）
        :param center: 导弹中心
        :param length_a: 导弹长度
        :param length_b: 导弹长度
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
