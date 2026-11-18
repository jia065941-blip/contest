# -*-coding:utf-8 -*-
from io import BytesIO
import math
import os
import sys
import threading

# 控制台留给仿真运行提示，渲染库导入时默认隐藏版本与社区问候。
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
from pygame.draw import rect
import typing


class Renderer:
    """
    实时渲染器 - 独立线程运行pygame窗口

    Attributes:
        MAP_W: 屏幕宽度（像素）
        MAP_H: 屏幕高度（像素）
    """

    BACKGROUND_COLOR = (244, 246, 249)
    GRID_COLOR = (222, 227, 233)
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
        21002: (135, 94, 0),
        9202: (112, 58, 180),
        9400: (180, 107, 16),
        9500: (100, 56, 169),
        9600: (20, 112, 66),
        24000: (174, 67, 28),
    }
    MARKER_BACKGROUND_RADIUS = 28
    EYE_MARKER_SIZE = 75
    EYE_BACKGROUND_RADIUS = 38
    EYE_EDGE_COLOR = (136, 24, 45)
    SWARM_MARKER_SIZE = 34
    SWARM_BACKGROUND_RADIUS = 20

    def __init__(self, x_min, x_max, y_min, y_max, fps=60):
        self.entities_data = {}
        self.red_area = []
        self.running = False
        self.thread = None
        self.MIN_WINDOW_SIZE = 500
        self.X_MIN, self.X_MAX = float(x_min), float(x_max)
        self.Y_MIN, self.Y_MAX = float(y_min), float(y_max)
        self.MAP_W, self.MAP_H = 1000, 800
        self._init_position_pixel_ratio()
        self._recalculate_position_boundary()

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
        self.show_coordinate = True

        # 投影转换
        self.fps = fps

        # 阵营底圆使用固定屏幕尺寸，缓存透明贴图供每帧复用。
        self._side_backgrounds = {}

        # ========== 鼠标位置 ==========
        self.mouse_x = 0
        self.mouse_y = 0

        self.radar_angle = 0  # 视野塔角度

        self.draw_radius = 8

    @classmethod
    def render(cls, state: dict) -> bytes:
        """将组织方物理态势绘制为单帧二维 PNG。"""
        # 控制节点保留在系统状态中，地图只绘制行动和感知实体。
        entities = {
            identifier: dict(data)
            for identifier, data in state["entities"].items()
            if data["type"] != 35000
        }
        area, deployment = state["map_area"], state["deployment_area"]
        points = [(area["xMin"], area["yMin"]), (area["xMax"], area["yMax"])]
        points.extend((data["pos"]["x"], data["pos"]["y"]) for data in entities.values())
        points.extend(tuple(point) for polygon in deployment["coordinates"] for point in polygon)
        if any(not math.isfinite(axis) for point in points for axis in point):
            raise ValueError("Render coordinates must be finite")

        # 按地图、部署区和实体位置取景，并留出图标边距与图例侧栏。
        width, height, margin, legend_width = 1000, 800, 80, 180
        west, east = min(p[0] for p in points), max(p[0] for p in points)
        south, north = min(p[1] for p in points), max(p[1] for p in points)
        scale = max((east - west) / (width - 2 * margin),
                    (north - south) / (height - 2 * margin), 1e-6)
        center_x, center_y = (west + east) / 2, (south + north) / 2
        renderer = cls(center_x - width * scale / 2, center_x + width * scale / 2,
                       center_y - height * scale / 2, center_y + height * scale / 2)
        fonts_initialized = pygame.font.get_init()
        try:
            if not fonts_initialized:
                pygame.font.init()
            renderer._init_fonts()
            surface = pygame.Surface((width + legend_width, height))
            surface.fill(renderer.BACKGROUND_COLOR)
            renderer._draw_coordinate_system(surface)
            frame = ((area["xMin"], area["yMin"]), (area["xMax"], area["yMin"]),
                     (area["xMax"], area["yMax"]), (area["xMin"], area["yMax"]))
            pygame.draw.polygon(surface, renderer.MUTED_TEXT_COLOR,
                                [renderer._pos_to_screen(*point) for point in frame], 1)
            for polygon in deployment["coordinates"]:
                pygame.draw.polygon(surface, renderer.SIDE_COLORS[0],
                                    [renderer._pos_to_screen(*point) for point in polygon], 2)
            renderer._draw_entities(surface, entities, show_all=True)
            renderer.draw_legend2(surface)
            output = BytesIO()
            pygame.image.save(surface, output, "preview.png")
            return output.getvalue()
        finally:
            if not fonts_initialized:
                pygame.font.quit()

    def _render_loop(self):
        """渲染主循环（在独立线程中运行）"""
        pygame.init()
        screen = pygame.display.set_mode((self.MAP_W, self.MAP_H), pygame.RESIZABLE)
        pygame.display.set_caption("仿真态势显示")
        clock = pygame.time.Clock()
        self._init_fonts()

        while self.running:
            # ========== 事件处理 ==========
            self._deal_event()

            # 获取最新态势
            entities = self.entities_data.copy()

            # ========== 中性底色；阵营颜色随各单位显示 ==========
            screen.fill(self.BACKGROUND_COLOR)

            # ========== 绘制坐标系（无限延伸，与背景绑定） ==========
            if self.show_coordinate:
                self._draw_coordinate_system(screen)

            # ========== 绘制红方区域多边形 ==========
            if len(self.red_area) >= 3:
                red_area_pixel = [list(self._pos_to_screen(x, y)) for x, y in self.red_area]
                pygame.draw.polygon(screen, self.SIDE_COLORS[0], red_area_pixel, 2)

            # ========== 绘制实体 ==========
            self._draw_entities(screen, entities)

            # ========== 右上角图例 ==========
            self.draw_legend2(screen)

            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()

    def _deal_event(self):
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
                    self.show_coordinate = not self.show_coordinate
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

                # 重新计算坐标边界，使地图中心重新对齐到新的窗口中心，
                # 避免窗口缩放时坐标轴相对屏幕中心发生偏移
                self._recalculate_position_boundary()

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
        :param entities: {entity_id: {"pos": {"x": , "y": , "z": }}}
        """
        self.entities_data = entities

    def update_area(self, red_area: list[list[float]]):
        """
        更新区域数据（非阻塞，只保留最新）
        :param red_area: 红方区域
        """
        self.red_area = red_area if red_area else []

    def _init_position_pixel_ratio(self):
        """Keep equal metres per pixel and a fixed map centre."""
        self.meter_per_pixel = max((self.X_MAX-self.X_MIN)/self.MAP_W,
                                   (self.Y_MAX-self.Y_MIN)/self.MAP_H, 1e-9)
        self.center_x = (self.X_MIN+self.X_MAX)/2
        self.center_y = (self.Y_MIN+self.Y_MAX)/2

    def _recalculate_position_boundary(self):
        self.X_MIN = self.center_x-self.meter_per_pixel*self.MAP_W/2
        self.X_MAX = self.center_x+self.meter_per_pixel*self.MAP_W/2
        self.Y_MIN = self.center_y-self.meter_per_pixel*self.MAP_H/2
        self.Y_MAX = self.center_y+self.meter_per_pixel*self.MAP_H/2

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

    def _pos_to_screen(self, x:float, y:float):
        """
        线性映射（带缩放和偏移）
        返回屏幕坐标，原点在屏幕中心
        """

        px = (x-self.center_x) / self.meter_per_pixel
        py = (y-self.center_y) / self.meter_per_pixel

        cx, cy = self.MAP_W // 2, self.MAP_H // 2

        # 世界坐标转屏幕坐标（原点在屏幕中心）
        sx = px * self.zoom + cx + self.offset_x
        sy = -py * self.zoom + cy + self.offset_y
        return int(sx), int(sy)

    def _screen_to_pos(self, sx:float, sy:float):
        """
        _pos_to_screen 的逆变换：
        屏幕坐标（原点在屏幕中心） -> 场景坐标（米）
        """

        if self.zoom == 0:
            return 0.0, 0.0

        cx, cy = self.MAP_W // 2, self.MAP_H // 2

        px = (sx - cx - self.offset_x) / self.zoom
        py = -(sy - cy - self.offset_y) / self.zoom

        # 世界像素坐标 -> 场景坐标（米），屏幕中心对应地图中心。
        x = self.center_x + px * self.meter_per_pixel
        y = self.center_y + py * self.meter_per_pixel
        return x, y

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

    def _draw_coordinate_system2(self, screen):
        """
        绘制无限延伸的坐标系（与背景绑定），内容包括：
        1、坐标轴
        2、网格线
        3、坐标数字
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
        grid_color = self.GRID_COLOR

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
        axis_color = (160, 88, 100)
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
        axis_color = (81, 104, 154)
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
            pygame.draw.circle(screen, self.MUTED_TEXT_COLOR, origin_screen, 8, 2)
            pygame.draw.circle(screen, self.MUTED_TEXT_COLOR, origin_screen, 3)
            origin_text = self.font.render(f"O", True, self.MUTED_TEXT_COLOR)
            screen.blit(origin_text, (origin_screen[0] + 12, origin_screen[1] - 8))

        # ========== 显示X轴和Y轴标签（跟随坐标系） ==========
        x_label = self.font.render("X", True, (160, 88, 100))
        y_label = self.font.render("Y", True, (81, 104, 154))

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
                pygame.draw.line(screen, self.MUTED_TEXT_COLOR,
                                 (sx, sy - 5), (sx, sy + 5), 1)
                if abs(x - cx) > 5:
                    offset = int(x - cx)
                    text = self.font.render(f"{offset}", True, self.MUTED_TEXT_COLOR)
                    screen.blit(text, (sx - 10, sy + 8))
            x += step

        # Y轴刻度
        y_start = (min_wy // step) * step
        y_end = (max_wy // step) * step

        y = y_start
        while y <= y_end:
            sx, sy = self._world_to_screen(cx, y)
            if -20 <= sx <= self.MAP_W + 20 and -20 <= sy <= self.MAP_H + 20:
                pygame.draw.line(screen, self.MUTED_TEXT_COLOR,
                                 (sx - 5, sy), (sx + 5, sy), 1)
                if abs(y - cy) > 5:
                    offset = int(cy - y)
                    text = self.font.render(f"{offset}", True, self.MUTED_TEXT_COLOR)
                    screen.blit(text, (sx + 8, sy - 6))
            y += step

    def _draw_coordinate_system(self, screen):
        """
        绘制无限延伸的坐标系（与背景绑定），内容包括：
        1、坐标轴
        2、网格线
        3、坐标数字
        """

        small_space = self.meter_per_pixel * 50 # 小间隔（米），用于绘制网格线
        big_space = small_space * 2 # 大间隔（米），用于绘制坐标轴和坐标数字

        # 计算当前屏幕上的绘制坐标系范围
        min_x, min_y = self._screen_to_pos(0, self.MAP_H)
        max_x, max_y = self._screen_to_pos(self.MAP_W, 0)

        min_x = self.round_away_from_zero(min_x, small_space)
        min_y = self.round_away_from_zero(min_y, small_space)

        grid_color = self.GRID_COLOR

        x = min_x
        while x < max_x:
            sx, _ = self._pos_to_screen(x, 0)
            pygame.draw.line(screen, grid_color, (sx, 0), (sx, self.MAP_H), 1)
            x += small_space

        y = min_y
        while y < max_y:
            _, sy = self._pos_to_screen(0, y)
            pygame.draw.line(screen, grid_color, (0, sy), (self.MAP_W, sy), 1)
            y += small_space


        if min_x < 0 < max_x:
            # 绘制 y 轴
            self._draw_axis_y(screen, min_y, max_y, big_space)

        if min_y < 0 < max_y:
            # 绘制 x 轴
            self._draw_axis_x(screen, min_x, max_x, big_space)

        return 0

    def _draw_axis_y(self, screen, min_y, max_y, space):
        arrow_size = 10

        # 轴（蓝色）- 垂直方向
        axis_color = (81, 104, 154)
        p1 = self._pos_to_screen(0, min_y)
        p2 = self._pos_to_screen(0, max_y)
        pygame.draw.line(screen, axis_color, p1, p2, 2)

        # 箭头（上端）
        arrow_x = self._pos_to_screen(0, 0)[0]
        if arrow_x < 0 or arrow_x > self.MAP_W:
            arrow_x = max(20, min(self.MAP_W - 20, arrow_x))
        pygame.draw.polygon(screen, axis_color, [
            (arrow_x - arrow_size // 2, 10),
            (arrow_x, 0),
            (arrow_x + arrow_size // 2, 10)
        ])

        # 刻度线
        y = self.round_away_from_zero(min_y, space)
        while y < max_y:
            sx, sy = self._pos_to_screen(0, y)
            pygame.draw.line(screen, self.MUTED_TEXT_COLOR, (sx - 5, sy), (sx + 5, sy), 1)

            text = self.font.render(f"{y}", True, self.MUTED_TEXT_COLOR)
            screen.blit(text, (sx + 8, sy - 6))

            y += space

    def _draw_axis_x(self, screen, min_x, max_x, space):
        arrow_size = 10

        # 轴（红色）- 水平方向
        axis_color = (160, 88, 100)
        p1 = self._pos_to_screen(min_x, 0)
        p2 = self._pos_to_screen(max_x, 0)
        pygame.draw.line(screen, axis_color, p1, p2, 2)

        # X轴箭头（右端）
        arrow_y = self._pos_to_screen(0, 0)[1]
        if arrow_y < 0 or arrow_y > self.MAP_H:
            arrow_y = max(20, min(self.MAP_H - 20, arrow_y))
        pygame.draw.polygon(screen, axis_color, [
            (self.MAP_W - 10, arrow_y - arrow_size // 2),
            (self.MAP_W, arrow_y),
            (self.MAP_W - 10, arrow_y + arrow_size // 2)
        ])

        # 刻度线
        x = self.round_away_from_zero(min_x, space)
        while x < max_x:
            sx, sy = self._pos_to_screen(x, 0)
            pygame.draw.line(screen, self.MUTED_TEXT_COLOR, (sx, sy - 5), (sx, sy + 5), 1)

            text = self.font.render(f"{x}", True, self.MUTED_TEXT_COLOR)
            screen.blit(text, (sx - 10, sy + 8))

            x += space

    @staticmethod
    def round_away_from_zero(x, mod = 1):
        if mod <= 0:
            raise ValueError("mod must be positive")

        if x >= 0:
            return math.ceil(x / mod) * mod
        else:
            return math.floor(x/mod)*mod

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
        surface.blit(self._side_backgrounds[key], (x - radius, y - radius))

    def _draw_entities(self, screen, entities, *, show_all=False):
        """先绘制范围和所有底圆，再绘制所有图标，防止底圆遮挡邻近单位。"""
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
            if not show_all and (not data.get("isVisible", False) or data.get("health", 0) <= 0):
                continue

            pos = data.get("pos", {})
            pos_x, pos_y = pos.get("x", 0), pos.get("y", 0)
            entity_type = data.get("type")
            if entity_type == 9202 and not show_all:
                # 天眼作为全局支援标记，固定在坐标轴原点；只调整显示位置。
                x, y = self._pos_to_screen(0, 0)
            else:
                x, y = self._pos_to_screen(pos_x, pos_y)
            if x < -50 or x > self.MAP_W + 50 or y < -50 or y > self.MAP_H + 50:
                # 超出屏幕范围
                continue
            drawer = marker_drawers.get(entity_type)
            visible_entities.append((data, x, y, drawer))

            # 绘制实体速度方向
            # self._draw_entity_vel(screen, data)

            # 哨卫巢视野范围属于地图辅助线，位于全部单位底圆和图标下方。
            if data.get("type") == 9600:
                pixel_radius = self.km_to_pixel_radius(pos_x, pos_y, 20)
                pygame.draw.circle(screen, (108, 155, 126), (x, y), pixel_radius * self.zoom, 1)

        for data, x, y, drawer in visible_entities:
            radius = self.MARKER_BACKGROUND_RADIUS if drawer else self.draw_radius + 4
            if data.get("type") == 9202:
                radius = self.EYE_BACKGROUND_RADIUS
            elif data.get("type") == 24000:
                radius = self.SWARM_BACKGROUND_RADIUS
            self._draw_side_background(screen, x, y, data.get("side"), radius)

        # 天眼独立强调边位于图标下方，外沿比其阵营底圆再大一圈。
        for data, x, y, _ in visible_entities:
            if data.get("type") == 9202:
                pygame.draw.circle(screen, self.EYE_EDGE_COLOR, (x, y),
                                   self.EYE_BACKGROUND_RADIUS + 4, 3)

        labels = []
        for data, x, y, drawer in visible_entities:
            if drawer:
                size = 50
                if data.get("type") == 9202:
                    size = self.EYE_MARKER_SIZE
                elif data.get("type") == 24000:
                    size = self.SWARM_MARKER_SIZE
                drawer(screen, x, y, size)
            else:
                color = self.SIDE_COLORS.get(data.get("side"), self.NEUTRAL_COLOR)
                pygame.draw.circle(screen, color, (x, y), self.draw_radius)
                name = data.get("nameChn", str(data.get("id", "")))
                text_surface = self.font.render(str(name), True, color)
                labels.append((text_surface, text_surface.get_rect(center=(x, y - 12))))
        for text_surface, text_rect in labels:
            screen.blit(text_surface, text_rect)

    def _init_fonts(self):
        """预览与实时窗口共用中文字体和字号。"""
        if sys.platform == 'win32':
            font_name = "simhei"
        else:
            font_name = "Noto Sans CJK SC"
        font_path = pygame.font.match_font(font_name)
        # WSL 可直接复用 Windows 字体，不依赖 Linux 的字体索引工具。
        windows_font = "/mnt/c/Windows/Fonts/simhei.ttf"
        if font_path is None and os.path.isfile(windows_font):
            font_path = windows_font
        self.font = pygame.font.Font(font_path, 10)
        self.font_legend = pygame.font.Font(font_path, 20)

    def km_to_pixel_radius(self, x: float, y: float, range_km: float) -> float:
        """Convert a distance in kilometres to pixels."""
        return range_km * 1000 / self.meter_per_pixel

    def _draw_entity_vel(self, screen, entity:dict):
        pos = entity.get("pos", {})
        pos_x, pos_y = pos.get("x", 0), pos.get("y", 0)

        vel = entity.get("vel", {})
        vel_x, vel_y = vel.get("x", 0), vel.get("y", 0)
        vel_x, vel_y = self._normalize_vector_length(vel_x, vel_y, 10000)

        x, y = self._pos_to_screen(pos_x, pos_y)
        m, n = self._pos_to_screen(pos_x + vel_x, pos_y + vel_y)

        pygame.draw.line(screen, (255, 0, 0), (x, y), (m, n), 2)

    @staticmethod
    def _normalize_vector_length(x, y, length):
        """
        将向量 (x, y) 的长度缩放到指定的 length，方向保持不变。
        零向量或 length <= 0 时返回零向量。
        """
        norm = math.hypot(x, y)
        if norm == 0 or length <= 0:
            return 0.0, 0.0

        scale = length / norm
        return x * scale, y * scale

    def draw_destroyer(self, surf, cx, cy, size=50):
        """毁灭者：六边形 + 中心圆点"""
        k = size / 128.0

        def T(x, y): return (cx + (x - 64) * k, cy + (y - 64) * k)

        hexp = [T(x, y) for x, y in self.get_poly_pts(64, 64, 40, 6)]
        pygame.draw.polygon(surf, self.TYPE_COLORS[21000], hexp, max(2, int(3 * k)))
        pygame.draw.circle(surf, self.TYPE_COLORS[21000], T(64, 64), max(1, int(9 * k)))

    def draw_assault(self, surf, cx, cy, size=50):
        """突击者：向上三角形 + 中心圆点"""
        k = size / 128.0

        def T(x, y): return (cx + (x - 64) * k, cy + (y - 64) * k)

        tri = [T(64, 20), T(104, 104), T(24, 104)]
        pygame.draw.polygon(surf, self.TYPE_COLORS[21001], tri, max(2, int(3 * k)))
        pygame.draw.circle(surf, self.TYPE_COLORS[21001], T(64, 76), max(1, int(9 * k)))

    def draw_scavenger(self, surf, cx, cy, size=50):
        """清道夫：圆环 + 交叉线"""
        k = size / 128.0

        def T(x, y): return (cx + (x - 64) * k, cy + (y - 64) * k)

        w = max(2, int(3 * k))
        pygame.draw.circle(surf, self.TYPE_COLORS[21002], T(64, 64), int(40 * k), w)
        pygame.draw.line(surf, self.TYPE_COLORS[21002], T(40, 40), T(88, 88), w)
        pygame.draw.line(surf, self.TYPE_COLORS[21002], T(88, 40), T(40, 88), w)

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
        """母巢：实心六边形 + 深色中心"""
        k = size / 128.0

        def T(x, y): return (cx + (x - 64) * k, cy + (y - 64) * k)

        hexp = [T(x, y) for x, y in self.get_poly_pts(64, 64, 40, 6)]
        pygame.draw.polygon(surf, self.TYPE_COLORS[9400], hexp)
        pygame.draw.circle(surf, (26, 18, 8), T(64, 64), max(1, int(12 * k)))

    def draw_sentry_hive(self, surf, cx, cy, size=50):
        """哨卫巢：五边形 + 中心圆点"""
        k = size / 128.0

        def T(x, y): return (cx + (x - 64) * k, cy + (y - 64) * k)

        pent = [T(x, y) for x, y in self.get_poly_pts(64, 64, 42, 5)]
        pygame.draw.polygon(surf, (226, 240, 231), pent)
        pygame.draw.polygon(surf, self.TYPE_COLORS[9600], pent, max(1, int(3 * k)))
        pygame.draw.circle(surf, self.TYPE_COLORS[9600], T(64, 64), max(1, int(10 * k)))

    def draw_dark_hive(self, surf, cx, cy, size=50):
        """暗巢：菱形 + 深色中心"""
        k = size / 128.0

        def T(x, y): return (cx + (x - 64) * k, cy + (y - 64) * k)

        rhomb = [T(64, 20), T(104, 64), T(64, 108), T(24, 64)]
        pygame.draw.polygon(surf, self.TYPE_COLORS[9500], rhomb)
        pygame.draw.circle(surf, (16, 11, 24), T(64, 64), max(1, int(10 * k)))

    def draw_swarm(self, surf, cx, cy, size=50):
        """自爆虫群：三个圆 + 中心小圆"""
        k = size / 128.0

        def T(x, y): return (cx + (x - 64) * k, cy + (y - 64) * k)

        r_big = max(1, int(22 * k))
        r_small = max(1, int(8 * k))
        pygame.draw.circle(surf, self.TYPE_COLORS[24000], T(64, 32), r_big, 1)
        pygame.draw.circle(surf, self.TYPE_COLORS[24000], T(36, 92), r_big, 1)
        pygame.draw.circle(surf, self.TYPE_COLORS[24000], T(92, 92), r_big, 1)
        pygame.draw.circle(surf, self.TYPE_COLORS[21002], T(64, 72), r_small)

    def draw_legend2(self, surface,
                     margin=10,  # 距离窗口边缘的边距
                     icon_size=32,  # 图例中图标的大小
                     item_h=38,  # 每个条目的高度
                     panel_w=160,  # 图例面板宽度
                     bg_alpha=240):  # 背景透明度

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
            text = self.font_legend.render(name, True, self.TEXT_COLOR)
            tx = x0 + 10 + icon_size + 10
            ty = y0 + 8 + i * item_h + (icon_size - text.get_height()) // 2
            surface.blit(text, (tx, ty))

        # 底圆说明与单位类型图例分开，避免将图标自身颜色解释为阵营。
        divider_y = y0 + n * item_h + 8
        pygame.draw.line(surface, self.PANEL_BORDER_COLOR,
                         (x0 + 10, divider_y), (x0 + panel_w - 10, divider_y))
        for index, (side, label) in enumerate(((0, "人族"), (1, "虫族"))):
            cx = x0 + 16 + index * (panel_w // 2)
            cy = divider_y + 22
            self._draw_side_background(surface, cx, cy, side, 10)
            text = self.font_legend.render(label, True, self.TEXT_COLOR)
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
