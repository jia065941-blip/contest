import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import numpy as np
import pygame
import imageio_ffmpeg
from .icons import Icons
from .rules import NAMES, WEIGHTS
from .typography import ReplayFont, font_files

FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
PROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def recorded_step(meta, simulation_seconds):
    step_seconds = meta.get('step_seconds', meta.get('audit', {}).get('step_seconds'))
    if (isinstance(step_seconds, bool) or not isinstance(step_seconds, (int, float))
            or not math.isfinite(step_seconds) or step_seconds <= 0):
        raise ValueError('静态帧需要日志记录的有效每步秒数')
    return simulation_seconds / step_seconds


class Timeline:
    def __init__(self, end, events, duration):
        self.intro, self.outro, self.duration = 2.5, 5.0, duration
        self.sim = np.linspace(0, end, int(end * 10) + 1)
        last_event = max((event['t'] for event in events), default=end)
        density = np.ones_like(self.sim)
        for event in events:
            # 平滑改变倍速，连续摧毁时累加阅读时间。
            scale = 2.7 if event['kind'] in ('satellite', 'objective_destroyed') else 1.5
            distance = np.abs(self.sim - event['t'])
            mask = distance < 2.0
            window = .5 * (1 + np.cos(math.pi * distance[mask] / 2))
            if event['kind'] == 'objective_destroyed':
                density[mask] += 2.7 * window
            else:
                density[mask] = np.maximum(density[mask], 1 + (scale - 1) * window)
        accumulated = np.concatenate(([0.0], np.cumsum((density[1:] + density[:-1]) * np.diff(self.sim) / 2)))
        self.fast_start = min(end, last_event + 12)
        cut = min(int(np.searchsorted(self.sim, self.fast_start)), len(self.sim) - 1)
        self.fast_start = float(self.sim[cut])
        available = duration - self.intro - self.outro
        if 0 < cut < len(self.sim) - 1:
            tail = min(3.0, duration * .03)
            self.video = np.empty_like(self.sim)
            self.video[:cut + 1] = self.intro + accumulated[:cut + 1] / accumulated[cut] * (available - tail)
            self.video[cut + 1:] = self.intro + available - tail + (self.sim[cut + 1:] - self.fast_start) / (end - self.fast_start) * tail
        else:
            self.video = self.intro + accumulated / accumulated[-1] * available
        # 末尾快进之前的全部内容统一再加快1.5倍，快进和结算保持原时长。
        normal_video = float(np.interp(self.fast_start, self.sim, self.video))
        self.normal_speed = 1.5
        self.normal_seconds_saved = normal_video * (1 - 1 / self.normal_speed)
        self.video = np.where(self.video <= normal_video, self.video / self.normal_speed,
                              self.video - self.normal_seconds_saved)
        self.intro /= self.normal_speed
        self.duration -= self.normal_seconds_saved

    def simulation_time(self, video_time):
        return float(np.interp(video_time, self.video, self.sim))

    def video_time(self, simulation_time):
        return float(np.interp(simulation_time, self.sim, self.video))


class ReplayPainter:
    BACKGROUND_COLOR = (12, 22, 39)
    MAP_COLOR = (20, 32, 56)
    PANEL_COLOR = (26, 40, 61)
    GRID_COLOR = (49, 72, 99)
    GRID_CROSS_COLOR = (67, 97, 125)
    SECONDARY_TEXT = (190, 207, 225)
    MUTED_TEXT = (180, 197, 216)
    EVENT_ROWS = 6
    EVENT_PITCH = 44
    EVENT_SLIDE_SECONDS = .35
    EVENT_FOLD_SECONDS = 1.0
    TIMELINE_MARKER_SIZE = 32
    TIMELINE_TRACK_Y = 1004
    MAP_FOOTER_Y = 900
    RED_SCALE = 1.95
    HIVE_SCALE = 1.5
    FONT_SCALE = 1.16
    ASSIGNMENT_COLOR = (255, 185, 92)
    HIT_COLORS = {0: (255, 126, 132), 1: (98, 212, 240)}
    RED_TRAIL_SECONDS = .3
    RED_TRAIL_WIDTHS = {21000: 9.0, 21001: 6.0, 21002: 6.0}
    RED_TRAIL_ALPHA = 80
    RED_RING_ALPHA = 64
    SATELLITE_FILL_ALPHA = 10
    REMNANT_COLOR = (51, 65, 85)
    REMNANT_SCALE = .6
    GAIN_BACKGROUND = (11, 25, 43, 232)
    GAIN_TEXT = (255, 216, 160)

    def __init__(self, meta, times, states, index, events, summary, timeline, font_file=None, latin_font_file=None):
        pygame.font.init()
        self.font_files = font_files(font_file, latin_font_file)
        self.screen = pygame.Surface((1920, 1080))
        self.meta, self.times, self.states, self.index = meta, times, states, index
        self.events, self.summary, self.timeline = events, summary, timeline
        sizes = (14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 29, 36, 44, 48, 52, 72, 88)
        self.fonts = {size: ReplayFont(*self.font_files, round(size * self.FONT_SCALE)) for size in sizes}
        self.bold = {size: ReplayFont(*self.font_files, round(size * self.FONT_SCALE), bold=True) for size in sizes}
        self.map_box = pygame.Rect(36, 148, 1350, 786)
        self.timeline_box = pygame.Rect(36, 950, 1848, 118)
        self.legend_box = pygame.Rect(638, 24, 748, 110)
        self.score_box = pygame.Rect(1410, 24, 474, 184)
        self.stats_box = pygame.Rect(1410, 238, 474, 696)
        self.clock_position = (380, 28)
        self.area = meta['map_area']
        width = self.area['xMax'] - self.area['xMin']
        height = self.area['yMax'] - self.area['yMin']
        if not all(math.isfinite(value) and value > 0 for value in (width, height)):
            raise ValueError('地图x/y边界须为有限且递增的米制坐标')
        scale = min((self.map_box.width - 80) / width, (self.map_box.height - 100) / height)
        self.pixels_per_meter = scale
        self.map_w, self.map_h = width * scale, height * scale
        self.map_left = self.map_box.centerx - self.map_w / 2
        self.map_top = self.map_box.centery - self.map_h / 2
        self.map_rect = pygame.Rect(round(self.map_left), round(self.map_top), round(self.map_w), round(self.map_h))
        self.objectives = {row['id']: row for row in summary['objectives']}
        self.total_weight = sum(row['weight'] for row in self.objectives.values())
        self.event_at = [timeline.video_time(event['t']) for event in events]
        self.damage = meta.get('damage', [])
        self.damage_at = [timeline.video_time(event['t']) for event in self.damage]
        self.dark_at = {event['target']: event['t'] for event in events if event['kind'] == 'dark_discovered'}
        self.destroy_at = {event['target']: event['t'] for event in events if event['kind'] == 'objective_destroyed'}
        self.units = [(int(key), data) for key, data in meta['entities'].items() if data['type'] in NAMES]
        self.headings = self.make_movement_headings()
        self.troop_counts = self.make_troop_counts()
        self.feed = self.make_event_feed()
        self.feed_at = [timeline.video_time(event['t']) for event in self.feed]
        self.feed_column_cache = {}
        self.feed_box = pygame.Rect(1424, 630, 446, self.EVENT_ROWS * self.EVENT_PITCH)
        self.feed_layer = pygame.Surface(self.feed_box.size, pygame.SRCALPHA)
        self.renderer = Icons()
        self.timeline_markers = self.make_timeline_markers()
        self.assignment_tracks = {}
        for row in meta.get('assignments', []):
            stamps, targets = self.assignment_tracks.setdefault(row['source'], ([], []))
            stamps.append(row['t'])
            targets.append(row['target'])
        self.guidance_tracks = {}
        for row in meta.get('guidance', []):
            stamps, positions = self.guidance_tracks.setdefault(row['source'], ([], []))
            stamps.append(row['t'])
            positions.append(row['position'])
        self.sprites = self.make_sprites()
        self.heading_sprites = {}
        self.remnants = self.make_remnants()
        self.score_layer = pygame.Surface((1920, 1080), pygame.SRCALPHA)
        self.score_glows = {}
        self.text_cache = {}
        self.static = self.make_static()
        self.static_frame = None

    def text(self, value, position, size=23, color=(226, 234, 246), bold=False, surface=None):
        key = (value, size, color, bold)
        if key not in self.text_cache:
            if len(self.text_cache) > 1800:
                self.text_cache.clear()
            font = self.bold[size] if bold else self.fonts[size]
            self.text_cache[key] = font.render(value, True, color)
        (surface if surface is not None else self.screen).blit(self.text_cache[key], position)

    def fitted_text(self, value, position, width, size=23, color=(226, 234, 246), bold=False, surface=None):
        font = self.bold[size] if bold else self.fonts[size]
        original = value
        while value and font.size(value)[0] > width:
            value = value[:-1]
        if value != original:
            value = value[:-1] + '…'
        self.text(value, position, size, color, bold, surface)

    def point(self, x, y):
        """将米制平面坐标映射为屏幕坐标，x向右、y向上。"""
        screen_x = self.map_left + (x - self.area['xMin']) * self.pixels_per_meter
        screen_y = self.map_top + (self.area['yMax'] - y) * self.pixels_per_meter
        return round(screen_x), round(screen_y)

    def grid_coordinates(self):
        """两轴使用相同的米制格距，保持网格的真实长宽比例。"""
        requested = max((self.area['xMax'] - self.area['xMin']) / 10,
                        (self.area['yMax'] - self.area['yMin']) / 7)
        magnitude = 10 ** math.floor(math.log10(requested))
        spacing = next(value * magnitude for value in (1, 2, 5, 10)
                       if value * magnitude >= requested)
        def ticks(low, high):
            start = (math.floor(low / spacing) + 1) * spacing
            return np.arange(start, high, spacing)
        return ticks(self.area['xMin'], self.area['xMax']), ticks(self.area['yMin'], self.area['yMax'])

    def make_sprites(self):
        functions = {21000: 'draw_destroyer', 21001: 'draw_assault', 21002: 'draw_scavenger',
                     24000: 'draw_swarm', 9400: 'draw_hive', 9500: 'draw_dark_hive', 9600: 'draw_sentry_hive'}
        sprites = {}
        for kind, function in functions.items():
            size = (48 if kind == 9400 else 50) if kind in WEIGHTS else (23 if kind == 21000 else 18)
            red = kind in (21000, 21001, 21002)
            if red:
                size = round(size * self.RED_SCALE)
            elif kind == 9400:
                size = round(size * self.HIVE_SCALE)
            sprite = pygame.Surface((size + 24, size + 24), pygame.SRCALPHA)
            center = (size + 24) // 2
            if red:
                radius = math.ceil(size * (.43 if kind == 21000 else .32)) + 2
                pygame.draw.circle(sprite, (255, 110, 113, self.RED_RING_ALPHA), (center, center), radius, 1)
            else:
                for radius, alpha in ((size // 2 + 7, 9), (size // 2 + 3, 18), (size // 2, 34)):
                    pygame.draw.circle(sprite, (65, 165, 230, alpha), (center, center), radius)
            getattr(self.renderer, function)(sprite, center, center, size)
            sprites[kind] = sprite
        muted = pygame.Surface(sprites[9500].get_size(), pygame.SRCALPHA)
        center = muted.get_width() // 2
        for x in range(center - 15, center + 16, 6):
            for y in range(center - 15, center + 16, 6):
                if abs(x - center) + abs(y - center) < 19:
                    pygame.draw.circle(muted, (111, 145, 177, 130), (x, y), 1)
        pygame.draw.polygon(muted, (149, 170, 192, 190), [(center, center - 14), (center + 14, center),
                                                       (center, center + 14), (center - 14, center)], 2)
        sprites['unknown_dark'] = muted
        return sprites

    def make_remnants(self):
        sprites = {}
        for kind in WEIGHTS:
            size = round((48 if kind == 9400 else 42) * .64 * self.REMNANT_SCALE)
            sprite = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.line(sprite, self.REMNANT_COLOR, (0, 0), (size - 1, size - 1), 2)
            pygame.draw.line(sprite, self.REMNANT_COLOR, (0, size - 1), (size - 1, 0), 2)
            sprites[kind] = sprite
        return sprites

    def make_static(self, static_frame=False):
        surface = pygame.Surface((1920, 1080))
        surface.fill(self.BACKGROUND_COLOR)
        pygame.draw.rect(surface, self.MAP_COLOR, self.map_box, border_radius=18)
        glow = pygame.Surface((1920, 1080), pygame.SRCALPHA)
        for inset in range(1, 11):
            alpha = round(28 * (1 - inset / 12)**2)
            pygame.draw.rect(glow, (56, 189, 248, alpha), self.map_box.inflate(-2 * inset, -2 * inset),
                             width=1, border_radius=max(1, 18 - inset))
        surface.blit(glow, (0, 0))
        pygame.draw.rect(surface, (61, 111, 145), self.map_box, width=1, border_radius=18)
        pygame.draw.rect(surface, self.PANEL_COLOR, self.score_box, border_radius=18)
        pygame.draw.rect(surface, self.PANEL_COLOR, self.stats_box, border_radius=18)
        pygame.draw.rect(surface, self.PANEL_COLOR, self.timeline_box, border_radius=15)
        self.fitted_text(self.meta['title'], (40, 15), 800, 52, bold=True, surface=surface)
        subtitle = self.meta['subtitle']
        if subtitle:
            self.fitted_text(subtitle, (56, self.MAP_FOOTER_Y), 1300, 20, self.SECONDARY_TEXT, surface=surface)
        pygame.draw.rect(surface, (64, 108, 142), self.map_rect, 1)
        self.text('x向东 · y向北 · 单位：米', (self.map_box.left + 20, self.map_box.top + 12),
                  16, self.SECONDARY_TEXT, surface=surface)
        grid_x, grid_y = self.grid_coordinates()
        for world_x in grid_x:
            x, _ = self.point(world_x, self.area['yMin'])
            pygame.draw.line(surface, self.GRID_COLOR, (x, self.map_rect.top), (x, self.map_rect.bottom))
        for world_y in grid_y:
            _, y = self.point(self.area['xMin'], world_y)
            pygame.draw.line(surface, self.GRID_COLOR, (self.map_rect.left, y), (self.map_rect.right, y))
            for world_x in grid_x:
                x, _ = self.point(world_x, world_y)
                pygame.draw.line(surface, self.GRID_CROSS_COLOR, (x - 3, y), (x + 3, y))
                pygame.draw.line(surface, self.GRID_CROSS_COLOR, (x, y - 3), (x, y + 3))
        tint = pygame.Surface((1920, 1080), pygame.SRCALPHA)
        for polygon in self.meta['red_area']:
            red_points = [self.point(*pair) for pair in polygon]
            region = pygame.Surface((1920, 1080), pygame.SRCALPHA)
            pygame.draw.polygon(region, (255, 255, 255, 255), red_points)
            stripes = pygame.Surface((1920, 1080), pygame.SRCALPHA)
            pygame.draw.polygon(stripes, (240, 98, 142, 16), red_points)
            for x in range(-1080, 1920, 18):
                pygame.draw.line(stripes, (240, 98, 142, 43), (x, 1080), (x + 1080, 0), 1)
            stripes.blit(region, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            tint.blit(stripes, (0, 0))
        surface.blit(tint, (0, 0))
        for polygon in self.meta['red_area']:
            pygame.draw.polygon(surface, (158, 75, 115), [self.point(*pair) for pair in polygon], 1)
        self.text('人族单位得分', (1440, 42), 26, bold=True, surface=surface)
        self.text('双方实时兵力', (1440, 258), 23, bold=True, surface=surface)
        self.text('移动单位', (1440, 292), 18, self.SECONDARY_TEXT, surface=surface)
        self.text('虫族建筑摧毁进度', (1440, 432), 23, bold=True, surface=surface)
        self.text('关键事件', (1440, 591), 26, bold=True, surface=surface)
        for y in (598, 606):
            pygame.draw.line(surface, self.SECONDARY_TEXT, (1817, y), (1833, y), 2)
        pygame.draw.line(surface, self.SECONDARY_TEXT, (1847, 596), (1847, 616), 2)
        pygame.draw.lines(surface, self.SECONDARY_TEXT, False, [(1840, 610), (1847, 617), (1854, 610)], 2)
        for row, kinds in enumerate(((21000, 21001, 21002), (9400, 9600, 9500, 24000))):
            for column, kind in enumerate(kinds):
                x, y = self.legend_box.left + 20 + column * 180, self.legend_box.top + 26 + row * 58
                function = {21000: 'draw_destroyer', 21001: 'draw_assault', 21002: 'draw_scavenger',
                            9400: 'draw_hive', 9600: 'draw_sentry_hive', 9500: 'draw_dark_hive',
                            24000: 'draw_swarm'}[kind]
                getattr(self.renderer, function)(surface, x + 12, y, 40)
                self.text(NAMES[kind], (x + 35, y - 13), 21, self.SECONDARY_TEXT, surface=surface)
        pygame.draw.line(surface, (61, 87, 115), (self.legend_box.left + 20, 83),
                         (self.legend_box.right - 20, 83), 1)
        x, y = self.legend_box.left + 20 + 3 * 180, self.legend_box.top + 26
        for offset in (0, 12, 24):
            pygame.draw.line(surface, self.ASSIGNMENT_COLOR, (x + offset, y), (x + offset + 7, y), 2)
        pygame.draw.circle(surface, self.ASSIGNMENT_COLOR, (x + 31, y), 5, 2)
        self.text('目标引导', (x + 44, y - 13), 21, self.SECONDARY_TEXT, surface=surface)
        return surface

    def render(self, video_t, *, simulation_seconds=None, static=False):
        if static and self.static_frame is None:
            self.static_frame = self.make_static(static_frame=True)
        self.screen.blit(self.static_frame if static else self.static, (0, 0))
        t = self.timeline.simulation_time(video_t) if simulation_seconds is None else simulation_seconds
        i = min(max(bisect_right(self.times, t) - 1, 0), len(self.times) - 1)
        current = self.states[i]
        positions, _ = self.sampled_positions(t)
        headings = self.movement_headings(t)
        destroyed = {key for key, at in self.destroy_at.items() if at <= t}
        score = 100 * sum(self.objectives[key]['weight'] for key in destroyed) / self.total_weight
        self.screen.set_clip(self.map_rect.inflate(4, 4))
        self.draw_trails(video_t)
        self.draw_assignments(t, positions, current)
        guidance_endpoints = self.draw_guidance(t, positions, current)
        for polygon in self.meta['red_area']:
            pulse = round(144 + 32 * math.sin(video_t * 2))
            pygame.draw.polygon(self.screen, (pulse, 82, 124), [self.point(*pair) for pair in polygon], 2)
        for event in self.events:
            if event['kind'] == 'objective_destroyed' and event['t'] <= t:
                x, y = self.point(*event['position'])
                sprite = self.remnants[event['type']]
                self.screen.blit(sprite, sprite.get_rect(center=(x, y)))
        focus = {event['target'] for event, at in zip(self.events, self.event_at) if 0 <= video_t - at < 1.55}
        focus.update(hit['target'] for hit, at in zip(self.damage, self.damage_at) if 0 <= video_t - at < .55)
        labels = []
        if static:
            # 静态图允许图标使用上方和两侧留白，底部预留真实场景信息。
            unit_clip = self.map_box.inflate(-8, -8)
            unit_clip.height = min(unit_clip.bottom, self.MAP_FOOTER_Y - 8) - unit_clip.top
            self.screen.set_clip(unit_clip)
        for key, data in sorted(self.units, key=lambda item: item[1]['type'] in WEIGHTS):
            row = current[self.index[key]]
            if not row[4] or row[3] <= 0 or key in destroyed:
                continue
            kind = data['type']
            x, y = self.point(*positions[self.index[key]])
            if kind == 44000:
                pygame.draw.circle(self.screen, (156, 169, 187), (x, y), 3, 1)
                continue
            sprite = self.sprites['unknown_dark' if kind == 9500 and t < self.dark_at.get(key, math.inf) else kind]
            if kind in (21000, 21001):
                angle = int(round(headings[self.index[key]])) % 360
                cache_key = (kind, angle)
                if cache_key not in self.heading_sprites:
                    self.heading_sprites[cache_key] = pygame.transform.rotate(sprite, angle)
                sprite = self.heading_sprites[cache_key]
            if kind in (9400, 9600):
                radius = round(29 * self.HIVE_SCALE) if kind == 9400 else 25
                color = self.renderer.TYPE_COLORS[kind]
                for segment in range(8):
                    angle = video_t * .4 + segment * math.tau / 8
                    start = (round(x + math.cos(angle) * radius), round(y + math.sin(angle) * radius))
                    end = (round(x + math.cos(angle) * (radius + 4)), round(y + math.sin(angle) * (radius + 4)))
                    pygame.draw.line(self.screen, color, start, end, 1)
            self.screen.blit(sprite, sprite.get_rect(center=(x, y)))
            if kind in WEIGHTS and key in focus:
                label = f'{NAMES[kind]} {key}'
                width = self.fonts[21].size(label)[0]
                for dx, dy in ((24, -42), (24, 22), (-width - 24, -42), (-width - 24, 22)):
                    rectangle = pygame.Rect(x + dx, y + dy, width, self.fonts[21].get_height())
                    if self.map_rect.contains(rectangle) and not any(rectangle.colliderect(old) for old in labels):
                        self.text(label, rectangle.topleft, 21, (225, 237, 248))
                        labels.append(rectangle)
                        break
            if kind in WEIGHTS:
                hp = max(0, min(row[3] / self.objectives[key]['initial_hp'], 1))
                bar_y = y + (34 if kind == 9400 else 25)
                pygame.draw.rect(self.screen, (28, 61, 79), (x - 16, bar_y, 32, 3))
                pygame.draw.rect(self.screen, (76, 212, 177), (x - 16, bar_y, round(32 * hp), 3))
        self.screen.set_clip(self.map_rect.inflate(4, 4))
        for endpoint in guidance_endpoints:
            pygame.draw.circle(self.screen, self.ASSIGNMENT_COLOR, endpoint, 6, 2)
        self.effects(video_t, t, positions)
        self.screen.set_clip(None)
        self.draw_score(score, video_t)
        self.text('/ 100', (1785, 140), 23, self.SECONDARY_TEXT)
        self.draw_troops(t)
        for n, kind in enumerate((9400, 9600, 9500)):
            total = sum(row['type'] == kind for row in self.objectives.values())
            count = sum(self.objectives[key]['type'] == kind for key in destroyed)
            y = 470 + n * 32
            self.text(NAMES[kind], (1440, y), 21)
            pygame.draw.rect(self.screen, (49, 66, 89), (1540, y + 9, 210, 10), border_radius=5)
            pygame.draw.rect(self.screen, (80, 185, 165), (1540, y + 9, round(210 * count / total) if total else 0, 10), border_radius=5)
            self.text(f'{count} / {total}', (1770, y), 21)
        self.event_feed(video_t, use_steps=static)
        if static:
            self.text(f'第{recorded_step(self.meta, t):g}步', self.clock_position, 48, bold=True)
        else:
            minute, second = divmod(int(t), 60)
            self.text(f'{minute:02}:{second:02}', self.clock_position, 48, bold=True)
            self.text(f'仿真时间 / {int(self.times[-1])} 秒', (self.clock_position[0], 96), 20, self.SECONDARY_TEXT)
        self.draw_timeline(video_t, t, use_steps=static)
        self.score_transfer(video_t, labels)
        if not static:
            self.draw_fast_forward(video_t, t)
        if not static and video_t < self.timeline.intro:
            details = ' · '.join(value for value in (self.meta['scenario_name'], self.meta['blue_name']) if value)
            self.banner('对局回放', details, min(1, video_t / .35))
        if not static and video_t >= self.timeline.duration - self.timeline.outro:
            remaining = len(self.objectives) - len(destroyed)
            ending = '正常结束' if self.summary['natural_done'] else '日志结束'
            progress = min(1, (video_t - self.timeline.duration + self.timeline.outro) / .9)
            self.banner('完整摧毁得分', f'{ending}  ·  剩余 {remaining} 个计分目标', 1.0, score=score, progress=progress)
        return self.screen

    def sampled_positions(self, t):
        i = min(max(bisect_right(self.times, t) - 1, 0), len(self.times) - 1)
        j = min(i + 1, len(self.times) - 1)
        fraction = (t - self.times[i]) / (self.times[j] - self.times[i]) if j != i else 0
        return self.states[i, :, :2] * (1 - fraction) + self.states[j, :, :2] * fraction, i

    def active_assignments(self, sim_t, current):
        rows = []
        for source, (stamps, targets) in self.assignment_tracks.items():
            at = bisect_right(stamps, sim_t) - 1
            if at < 0 or targets[at] is None:
                continue
            target = targets[at]
            if any(current[self.index[key], 3] <= 0 or not current[self.index[key], 4] for key in (source, target)):
                continue
            if self.meta['entities'][str(target)]['type'] == 9500 and sim_t < self.dark_at.get(target, math.inf):
                continue
            rows.append((source, target))
        return rows

    def draw_assignments(self, sim_t, positions, current):
        layer = pygame.Surface(self.map_rect.size, pygame.SRCALPHA)
        for source, target in self.active_assignments(sim_t, current):
            start = self.point(*positions[self.index[source]])
            end = self.point(*positions[self.index[target]])
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length < 15:
                continue
            ux, uy = dx / length, dy / length
            ox, oy = start[0] - self.map_rect.left, start[1] - self.map_rect.top
            for distance in range(0, int(length - 8), 18):
                end_distance = min(distance + 9, length - 8)
                a = (ox + ux * distance, oy + uy * distance)
                b = (ox + ux * end_distance, oy + uy * end_distance)
                pygame.draw.line(layer, (*self.ASSIGNMENT_COLOR, 92), a, b, 1)
            tx, ty = ox + ux * (length - 9), oy + uy * (length - 9)
            pygame.draw.lines(layer, (*self.ASSIGNMENT_COLOR, 125), False,
                              [(tx - ux * 7 - uy * 4, ty - uy * 7 + ux * 4), (tx, ty),
                               (tx - ux * 7 + uy * 4, ty - uy * 7 - ux * 4)], 1)
        self.screen.blit(layer, self.map_rect.topleft)

    def active_guidance(self, sim_t, current):
        """仅显示存活且可见单位的真实航点，不将坐标匹配成实体目标。"""
        rows = []
        for source, (stamps, positions) in self.guidance_tracks.items():
            at = bisect_right(stamps, sim_t) - 1
            if at < 0 or positions[at] is None:
                continue
            row = current[self.index[source]]
            if row[3] > 0 and row[4]:
                rows.append((source, positions[at]))
        return rows

    def draw_guidance(self, sim_t, positions, current):
        layer = pygame.Surface(self.map_rect.size, pygame.SRCALPHA)
        endpoints = set()
        for source, target in self.active_guidance(sim_t, current):
            start = self.point(*positions[self.index[source]])
            end = self.point(*target)
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length < 15:
                continue
            endpoints.add(end)
            ux, uy = dx / length, dy / length
            ox, oy = start[0] - self.map_rect.left, start[1] - self.map_rect.top
            # 单位图标下方绘制半透明虚线；共享航点只绘制一个端点标记。
            for distance in range(18, int(length - 9), 24):
                finish = min(distance + 13, length - 9)
                pygame.draw.line(layer, (*self.ASSIGNMENT_COLOR, 145),
                                 (ox + ux * distance, oy + uy * distance),
                                 (ox + ux * finish, oy + uy * finish), 2)
            tx, ty = ox + ux * (length - 9), oy + uy * (length - 9)
            pygame.draw.lines(layer, (*self.ASSIGNMENT_COLOR, 185), False,
                              [(tx - ux * 9 - uy * 5, ty - uy * 9 + ux * 5), (tx, ty),
                               (tx - ux * 9 + uy * 5, ty - uy * 9 - ux * 5)], 2)
        self.screen.blit(layer, self.map_rect.topleft)
        return endpoints

    def make_movement_headings(self):
        # 提前计算日志位移方向；静止时保留最近一次方向，不推断攻击目标。
        result = np.zeros(self.states.shape[:2], dtype=np.float64)
        previous = np.zeros(self.states.shape[1], dtype=np.float64)
        for i in range(len(self.times) - 1):
            delta = self.states[i + 1, :, :2] - self.states[i, :, :2]
            dx = delta[:, 0] * self.pixels_per_meter
            dy = -delta[:, 1] * self.pixels_per_meter
            moving = np.any(np.abs(delta) >= 1e-10, axis=1)
            angles = np.degrees(np.arctan2(-dx, -dy))
            previous = np.where(moving, angles, previous)
            result[i] = previous
        result[-1] = previous
        return result

    def movement_headings(self, t):
        i = min(max(bisect_right(self.times, t) - 1, 0), len(self.times) - 1)
        j = min(i + 1, len(self.times) - 1)
        fraction = (t - self.times[i]) / (self.times[j] - self.times[i]) if j != i else 0
        difference = (self.headings[j] - self.headings[i] + 180) % 360 - 180
        return self.headings[i] + fraction * difference

    def make_troop_counts(self):
        groups = ((0, (21000, 21001, 21002)), (1, (24000,)))
        result = np.zeros((len(self.times), len(groups), 3), dtype=np.int32)
        for group, (side, kinds) in enumerate(groups):
            rows = [self.index[int(key)] for key, entity in self.meta['entities'].items()
                    if entity['side'] == side and entity['type'] in kinds]
            living = self.states[:, rows, 3] > 0
            deployed = np.maximum.accumulate(living, axis=0).sum(axis=1)
            alive = living.sum(axis=1)
            result[:, group, :] = np.column_stack((alive, deployed - alive, deployed))
        return result

    def troop_statistics(self, sim_t):
        i = min(max(bisect_right(self.times, sim_t) - 1, 0), len(self.times) - 1)
        return self.troop_counts[i]

    def draw_troops(self, sim_t):
        for i, ((alive, lost, deployed), label, color) in enumerate(zip(self.troop_statistics(sim_t),
                ('人族单位', '虫族单位'), ((255, 126, 132), (98, 212, 240)))):
            y = 324 + i * 44
            self.text(label, (1440, y), 21, color, True)
            self.text(f'存活 {alive}', (1560, y), 21)
            self.text(f'损失 {lost}', (1710, y), 21, self.SECONDARY_TEXT)
            pygame.draw.rect(self.screen, (55, 72, 94), (1440, y + 31, 414, 5), border_radius=2)
            width = round(414 * alive / deployed) if deployed else 0
            if width:
                pygame.draw.rect(self.screen, color, (1440, y + 31, width, 5), border_radius=2)

    def trail_segments(self, video_t):
        red_history = [self.sampled_positions(self.timeline.simulation_time(max(0, video_t - self.RED_TRAIL_SECONDS * (1 - k / 7)))) for k in range(8)]
        blue_history = [self.sampled_positions(self.timeline.simulation_time(max(0, video_t - .85 * (1 - k / 7)))) for k in range(8)]
        for key, data in self.units:
            if data['type'] not in (21000, 21001, 21002, 24000):
                continue
            index = self.index[key]
            color = self.renderer.TYPE_COLORS[data['type']]
            red = data['type'] in (21000, 21001, 21002)
            history = red_history if red else blue_history
            current = history[-1][1]
            if not self.states[current, index, 4] or self.states[current, index, 3] <= 0:
                continue
            for k in range(len(history) - 1):
                first, last = history[k], history[k + 1]
                if not self.states[first[1], index, 4] or not self.states[last[1], index, 4]:
                    continue
                start = self.point(*first[0][index])
                end = self.point(*last[0][index])
                if start == end:
                    continue
                previous_fraction = k / (len(history) - 1)
                fraction = (k + 1) / (len(history) - 1)
                widths = (self.RED_TRAIL_WIDTHS[data['type']] * previous_fraction**2,
                          self.RED_TRAIL_WIDTHS[data['type']] * fraction**2) if red else (1, 1)
                alpha = round(self.RED_TRAIL_ALPHA * fraction**2) if red else 25 + k * 20
                yield start, end, color, widths, alpha, red

    def draw_trails(self, video_t):
        layer = pygame.Surface(self.map_rect.size, pygame.SRCALPHA)
        for start, end, color, widths, alpha, red in self.trail_segments(video_t):
            start = (start[0] - self.map_rect.x, start[1] - self.map_rect.y)
            end = (end[0] - self.map_rect.x, end[1] - self.map_rect.y)
            if red:
                dx, dy = end[0] - start[0], end[1] - start[1]
                length = math.hypot(dx, dy)
                nx, ny = -dy / length, dx / length
                polygon = [(start[0] + nx * widths[0] / 2, start[1] + ny * widths[0] / 2),
                           (end[0] + nx * widths[1] / 2, end[1] + ny * widths[1] / 2),
                           (end[0] - nx * widths[1] / 2, end[1] - ny * widths[1] / 2),
                           (start[0] - nx * widths[0] / 2, start[1] - ny * widths[0] / 2)]
                pygame.draw.polygon(layer, (*color, alpha), polygon)
            else:
                pygame.draw.line(layer, (*color, alpha), start, end, 1)
        self.screen.blit(layer, self.map_rect.topleft)

    def effects(self, video_t, sim_t, positions):
        overlay = pygame.Surface((self.map_rect.width, self.map_rect.height), pygame.SRCALPHA)
        for event, at in zip(self.events, self.event_at):
            age = video_t - at
            if event['kind'] == 'satellite' and 0 <= age < 3.0:
                alpha = round(self.SATELLITE_FILL_ALPHA * (1 - age / 3))
                overlay.fill((128, 108, 247, alpha))
                radius = round(max(self.map_rect.size) * (age / 3))
                pygame.draw.circle(overlay, (156, 128, 255, 190), (overlay.get_width() // 2, overlay.get_height() // 2), max(1, radius), 2)
                scan_x = round(overlay.get_width() * (age / 3))
                pygame.draw.rect(overlay, (144, 126, 255, self.SATELLITE_FILL_ALPHA), (scan_x - 18, 0, 36, overlay.get_height()))
                center = (overlay.get_width() // 2, overlay.get_height() // 2)
                angle = age * math.tau / 3
                reach = max(1, radius)
                fan = [center, (round(center[0] + math.cos(angle) * reach), round(center[1] + math.sin(angle) * reach)),
                       (round(center[0] + math.cos(angle + .32) * reach), round(center[1] + math.sin(angle + .32) * reach))]
                pygame.draw.polygon(overlay, (144, 126, 255, self.SATELLITE_FILL_ALPHA), fan)
                for edge in fan[1:]:
                    pygame.draw.line(overlay, (176, 151, 255, 170), center, edge, 1)
                for segment in range(36):
                    tick_angle = segment * math.tau / 36
                    inner = (round(center[0] + math.cos(tick_angle) * max(0, radius - 5)),
                             round(center[1] + math.sin(tick_angle) * max(0, radius - 5)))
                    outer = (round(center[0] + math.cos(tick_angle) * radius), round(center[1] + math.sin(tick_angle) * radius))
                    pygame.draw.line(overlay, (190, 170, 255, 190), inner, outer, 1)
            if not 0 <= age < 1.55 or event['kind'] == 'satellite':
                continue
            point = event['position'] if event['kind'] != 'high_red_discovered' else positions[self.index[event['target']]]
            x, y = self.point(*point)
            center = (x - self.map_rect.left, y - self.map_rect.top)
            progress = age / 1.55
            fade = round(220 * (1 - progress))
            if event['kind'] == 'objective_destroyed':
                pygame.draw.circle(overlay, (255, 137, 53, fade), center, round(15 + 65 * progress), 4)
                pygame.draw.circle(overlay, (255, 220, 170, fade), center, round(8 + 30 * progress), 2)
                if age < .18:
                    pygame.draw.circle(overlay, (255, 249, 230, round(240 * (1 - age / .18))), center, max(2, round(24 * (1 - age / .18))))
                for k in range(4):
                    offset = round(math.sin(age * 70 + k * 2) * 15)
                    pygame.draw.rect(overlay, (82, 218, 245, round(fade * .4)), (center[0] - 24 + offset, center[1] - 18 + k * 12, 48, 2))
                for k in range(10):
                    angle = k * math.tau / 10
                    radius = 14 + 48 * progress
                    end = (round(center[0] + math.cos(angle) * radius), round(center[1] + math.sin(angle) * radius))
                    pygame.draw.circle(overlay, (255, 171, 61, fade), end, max(1, round(4 * (1 - progress))))
            elif event['kind'] == 'dark_discovered':
                color = self.renderer.TYPE_COLORS[9500]
                pygame.draw.circle(overlay, (*color, fade), center, round(22 + progress * 34), 3)
                pygame.draw.rect(overlay, (*color, fade), (center[0] - 21, center[1] - 21, 42, 42), 2)
            else:
                radius = round(19 + progress * 12)
                for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    p = (center[0] + sx * radius, center[1] + sy * radius)
                    pygame.draw.line(overlay, (35, 136, 219, fade), p, (p[0] - sx * 10, p[1]), 3)
                    pygame.draw.line(overlay, (35, 136, 219, fade), p, (p[0], p[1] - sy * 10), 3)
        self.screen.blit(overlay, self.map_rect.topleft)
        for hit, at in zip(self.damage, self.damage_at):
            if 0 <= video_t - at < .38:
                actual_positions, _ = self.sampled_positions(hit['t'])
                start = self.point(*actual_positions[self.index[hit['source']]])
                end = self.point(*actual_positions[self.index[hit['target']]])
                color = self.HIT_COLORS[self.meta['entities'][str(hit['source'])]['side']]
                pygame.draw.line(self.screen, color, start, end, 2)
                pygame.draw.circle(self.screen, color, end, 7, 2)
        satellite = next((event for event, at in zip(self.events, self.event_at) if event['kind'] == 'satellite' and
                          ((event['duration'] is not None and event['t'] <= sim_t < event['t'] + event['duration']) or
                           (event['duration'] is None and 0 <= video_t - at < 3))), None)
        if satellite is not None and not self.timeline.fast_start <= sim_t < self.times[-1]:
            value = '天眼启动'
            if satellite['duration'] is not None:
                remaining = math.ceil(satellite['t'] + satellite['duration'] - sim_t)
                value = f'天眼扫描中 · 剩余 {remaining} 秒'
            badge = self.map_notice_rectangle(self.fonts[23].size(value)[0] + 36, 43)
            pygame.draw.rect(self.screen, (111, 79, 190), badge, border_radius=10)
            self.text(value, (badge.x + 18, badge.y + 5), 23)

    def draw_score(self, score, video_t):
        pulse = 0.0
        for event, at in zip(self.events, self.event_at):
            elapsed = video_t - at - 1.25
            if event['kind'] == 'objective_destroyed' and 0 <= elapsed < .55:
                pulse = max(pulse, math.sin(math.pi * elapsed / .55))
        value = f'{score:05.2f}'
        if pulse > 0:
            self.score_layer.fill((0, 0, 0, 0))
            pygame.draw.rect(self.score_layer, (255, 193, 125, round(23 * pulse)),
                             (self.score_box.left + 12, self.score_box.top + 59, 440, 100), border_radius=18)
            self.screen.blit(self.score_layer, (0, 0))
            if value not in self.score_glows:
                glyph = self.bold[72].render(value, True, (255, 210, 151))
                blurred = pygame.transform.smoothscale(glyph, (max(1, glyph.get_width() // 6), max(1, glyph.get_height() // 6)))
                self.score_glows[value] = pygame.transform.smoothscale(blurred, (glyph.get_width() + 16, glyph.get_height() + 14))
            glow = self.score_glows[value]
            glow.set_alpha(round(110 * pulse))
            self.screen.blit(glow, (self.score_box.left + 17, self.score_box.top + 53))
        base, peak = (255, 163, 169), (255, 228, 177)
        color = tuple(round(start + (end - start) * pulse) for start, end in zip(base, peak))
        self.text(value, (self.score_box.left + 25, self.score_box.top + 60), 72, color, bold=True)

    def layout_score_capsules(self, video_t, labels):
        occupied = list(labels)
        placements = []
        bounds = self.map_rect.inflate(-8, -8)
        for event, at in zip(self.events, self.event_at):
            age = video_t - at
            if event['kind'] != 'objective_destroyed' or not 0 <= age < 1.1:
                continue
            x, y = self.point(*event['position'])
            gain = 100 * self.objectives[event['target']]['weight'] / self.total_weight
            value = f'+{gain:.2f}分'
            text_width, text_height = self.fonts[23].size(value)
            width, height = text_width + 24, text_height + 10
            anchor = pygame.Rect(x - width // 2, y - 94 - round(age * 18), width, height).clamp(bounds)
            rectangle = None
            # 每次上移24像素，受边界限制后再搜索下方及相邻列，绝不接受重叠候选。
            columns = [0]
            for offset in range(width + 12, bounds.width + width, width + 12):
                columns.extend((offset, -offset))
            rows = list(range(anchor.y, bounds.top - 1, -24)) + list(range(anchor.y + 24, bounds.bottom - height + 1, 24))
            rows.extend((bounds.top, bounds.bottom - height))
            for dx in columns:
                px = min(max(anchor.x + dx, bounds.left), bounds.right - width)
                for py in rows:
                    candidate = pygame.Rect(px, py, width, height)
                    if not any(candidate.inflate(6, 6).colliderect(old) for old in occupied):
                        rectangle = candidate
                        break
                if rectangle is not None:
                    break
            if rectangle is None:
                raise ValueError('地图内没有足够空间放置得分胶囊，请增大输出地图区域')
            occupied.append(rectangle)
            placements.append((event, value, rectangle))
        return placements

    def score_transfer(self, video_t, labels):
        capsules = self.layout_score_capsules(video_t, labels)
        self.frame_annotations = {'labels': tuple(labels), 'capsules': tuple(item[2] for item in capsules)}
        self.score_layer.fill((0, 0, 0, 0))
        for event, value, rectangle in capsules:
            pygame.draw.rect(self.score_layer, self.GAIN_BACKGROUND, rectangle, border_radius=rectangle.height // 2)
            pygame.draw.rect(self.score_layer, (158, 182, 204, 150), rectangle, 1, border_radius=rectangle.height // 2)
            glyph = self.fonts[23].render(value, True, self.GAIN_TEXT)
            self.score_layer.blit(glyph, (rectangle.x + 12, rectangle.y + 5))
        for event, at in zip(self.events, self.event_at):
            age = video_t - at
            if event['kind'] != 'objective_destroyed' or not 0 <= age < 1.6:
                continue
            x, y = self.point(*event['position'])
            linear = min(1, age / 1.25)
            fraction = linear * linear * (3 - 2 * linear)
            end = (self.score_box.centerx, self.score_box.top + 108)
            control = ((x + end[0]) / 2 - 80, max(self.score_box.top + 20, min(y, end[1]) - 140))

            def point_at(progress):
                return tuple(round((1 - progress)**2 * p0 + 2 * (1 - progress) * progress * pc + progress**2 * p1)
                             for p0, pc, p1 in zip((x, y), control, end))

            previous = point_at(max(0, fraction - .14))
            for segment in range(1, 9):
                progress = max(0, fraction - .14 + .14 * segment / 8)
                point = point_at(progress)
                pygame.draw.line(self.score_layer, (255, 211, 153, 20 + segment * 20), previous, point, 2)
                previous = point
            if age < 1.4:
                point = point_at(fraction)
                pygame.draw.circle(self.score_layer, (255, 204, 141, 35), point, 9)
                pygame.draw.circle(self.score_layer, (255, 220, 166, 90), point, 6)
                pygame.draw.circle(self.score_layer, (255, 231, 193, 235), point, 3)
        self.screen.blit(self.score_layer, (0, 0))

    def make_event_feed(self):
        grouped, rows = {}, []
        for event in self.events:
            if event['kind'] == 'high_red_discovered':
                continue
            marker = (event['kind'], math.floor(event['t'] / self.EVENT_FOLD_SECONDS))
            member = (event['kind'], event['target'], event['t'])
            if event['kind'] == 'dark_discovered' and marker in grouped:
                row = grouped[marker]
                row['members'].append(member)
                row['count'] += 1
                # 折叠条目在最后一个成员真实发生后显示，避免提前展示发现数量。
                row['t'] = max(row['t'], event['t'])
            else:
                row = {**event, 'members': [member], 'count': 1}
                rows.append(row)
                if event['kind'] == 'dark_discovered':
                    grouped[marker] = row
        return sorted(rows, key=lambda row: (row['t'], row['kind'], row['target']))

    def event_feed_layout(self, video_t):
        count = bisect_right(self.feed_at, video_t)
        rows = []
        for slot, i in enumerate(range(count - 1, max(-1, count - self.EVENT_ROWS - 1), -1)):
            y = self.feed_box.top + slot * self.EVENT_PITCH
            rectangle = pygame.Rect(1428, y, 438, self.EVENT_PITCH - 2)
            rows.append((self.feed[i], rectangle))
        return rows

    def event_stamp(self, simulation_seconds, use_steps=False):
        if use_steps:
            return f'第{recorded_step(self.meta, simulation_seconds):g}步'
        minute, second = divmod(int(simulation_seconds), 60)
        return f'{minute:02}:{second:02}'

    def event_feed_columns(self, use_steps=False):
        if use_steps not in self.feed_column_cache:
            stamp_width = max((self.fonts[19].size(self.event_stamp(event['t'], use_steps))[0]
                               for event in self.feed), default=0)
            stamp_right = 10 + stamp_width
            icon_center = stamp_right + 10 + 11
            self.feed_column_cache[use_steps] = (stamp_right, icon_center, icon_center + 11 + 12)
        return self.feed_column_cache[use_steps]

    def event_feed(self, video_t, *, use_steps=False):
        self.feed_layer.fill((0, 0, 0, 0))
        _, icon_center, text_left = self.event_feed_columns(use_steps)
        for slot, (event, rectangle) in enumerate(self.event_feed_layout(video_t)):
            card = pygame.Surface(rectangle.size, pygame.SRCALPHA)
            mother = event['kind'] == 'objective_destroyed' and event['type'] == 9400
            background = (65, 47, 39) if mother else (32, 49, 70)
            pygame.draw.rect(card, background, card.get_rect(), border_radius=7)
            if mother:
                pygame.draw.rect(card, (230, 165, 103), card.get_rect(), 1, border_radius=7)
            age = max(0, video_t - self.timeline.video_time(event['t']))
            fraction = min(1, age / self.EVENT_SLIDE_SECONDS)
            fraction = fraction * fraction * (3 - 2 * fraction)
            shift = round(12 * (1 - fraction)) if slot == 0 else 0
            self.text(self.event_stamp(event['t'], use_steps), (10, 9), 19, self.SECONDARY_TEXT, surface=card)
            self.renderer.event_marker(card, (icon_center, 21), 22, event)
            if event['kind'] == 'objective_destroyed':
                gain = 100 * WEIGHTS[event['type']] / self.total_weight
                line = f"{NAMES[event['type']]}摧毁"
                value = f'+{gain:.2f}分'
                gain_left = card.get_width() - 10 - self.fonts[21].size(value)[0]
                self.fitted_text(line, (text_left + shift, 5), gain_left - 12 - text_left - shift,
                                 23, (255, 226, 189) if mother else (230, 239, 248), surface=card)
                self.text(value, (gain_left, 8), 21, (255, 191, 133), surface=card)
            elif event['kind'] == 'dark_discovered':
                value = '发现暗巢' + (f" ×{event['count']}" if event['count'] > 1 else '')
                self.text(value, (text_left + shift, 5), 23, self.renderer.TYPE_COLORS[9500], surface=card)
            else:
                label = ('天眼探测' if event.get('confirmation') == 'native_satellite_detection_records'
                         else '天眼启动')
                self.text(label, (text_left + shift, 5), 23, (213, 197, 255), surface=card)
            self.feed_layer.blit(card, (rectangle.x - self.feed_box.x, rectangle.y - self.feed_box.y))
        self.screen.blit(self.feed_layer, self.feed_box.topleft)

    def make_timeline_markers(self, use_steps=False):
        end = self.times[-1] if use_steps else self.timeline.duration
        return [(event, (round(54 + 1810 * (event['t'] if use_steps else self.timeline.video_time(event['t'])) / end),
                         self.TIMELINE_TRACK_Y)) for event in self.events
                if event['kind'] in ('objective_destroyed', 'dark_discovered')]

    def draw_timeline(self, video_t, sim_t, *, use_steps=False):
        x0, y, length = 54, self.TIMELINE_TRACK_Y, 1810
        pygame.draw.line(self.screen, (75, 99, 125), (x0, y), (x0 + length, y), 5)
        markers = self.make_timeline_markers(use_steps=True) if use_steps else self.timeline_markers
        progress = sim_t / self.times[-1] if use_steps else video_t / self.timeline.duration
        x = round(x0 + length * progress)
        pygame.draw.circle(self.screen, (237, 243, 250), (x, y), 8)
        for event, center in markers:
            self.renderer.event_marker(self.screen, center, self.TIMELINE_MARKER_SIZE, event, filled=False)
        label = (f'{recorded_step(self.meta, sim_t):g} / {recorded_step(self.meta, self.times[-1]):g} 步'
                 if use_steps else f'播放 {video_t:05.1f} / {self.timeline.duration:.0f} 秒')
        self.text(label, (1864 - self.fonts[21].size(label)[0], 953), 21, self.SECONDARY_TEXT)
        legend = [('母巢摧毁', {'kind': 'objective_destroyed', 'type': 9400}),
                  ('哨卫巢摧毁', {'kind': 'objective_destroyed', 'type': 9600}),
                  ('暗巢摧毁', {'kind': 'objective_destroyed', 'type': 9500}),
                  ('暗巢发现', {'kind': 'dark_discovered'})]
        for i, (label, event) in enumerate(legend):
            x = 60 + i * 440
            self.renderer.event_marker(self.screen, (x + 14, 1040), self.TIMELINE_MARKER_SIZE, event, filled=False)
            self.text(label, (x + 44, 1026), 20, self.SECONDARY_TEXT)

    def map_notice_rectangle(self, width, height):
        rectangle = pygame.Rect(0, self.map_rect.top + 12, width, height)
        rectangle.centerx = self.map_box.centerx
        return rectangle

    def draw_fast_forward(self, video_t, sim_t):
        if not self.timeline.fast_start <= sim_t < self.times[-1]:
            return
        speed = (self.timeline.simulation_time(min(video_t + .1, self.timeline.duration)) - sim_t) / .1
        label = f'后续无新增关键事件 · 快进至日志终点  {speed:.1f}x'
        width = self.fonts[23].size(label)[0] + 116
        badge = self.map_notice_rectangle(width, 48)
        pygame.draw.rect(self.screen, (27, 49, 72), badge, border_radius=12)
        pulse = (math.sin(video_t * math.tau * 1.5) + 1) / 2
        color = tuple(round(low + (high - low) * pulse) for low, high in zip((146, 177, 192), (255, 211, 151)))
        pygame.draw.rect(self.screen, color, badge, 1, border_radius=12)
        for offset in (0, 10, 20):
            x = badge.x + 13 + offset
            arrow = ((x, badge.y + 16), (x + 7, badge.y + 24), (x, badge.y + 32))
            pygame.draw.lines(self.screen, color, False, arrow, 2)
            reflected = [(2 * badge.centerx - px, py) for px, py in arrow]
            pygame.draw.lines(self.screen, color, False, reflected, 2)
        self.text(label, (badge.x + 58, badge.y + 8), 23, (234, 224, 206))

    def banner(self, title, subtitle, opacity, score=None, progress=1):
        rectangle = pygame.Rect(330, 404, 760, 270)
        background = self.screen.subsurface(rectangle).copy()
        blurred = pygame.transform.smoothscale(pygame.transform.smoothscale(background, (76, 23)), rectangle.size)
        self.screen.blit(blurred, rectangle)
        panel = pygame.Surface(rectangle.size, pygame.SRCALPHA)
        polygon = [(0, 20), (20, 0), (730, 0), (760, 30), (760, 250), (740, 270), (20, 270), (0, 250)]
        pygame.draw.polygon(panel, (12, 31, 51, round(239 * opacity)), polygon)
        pygame.draw.polygon(panel, (80, 186, 213, round(210 * opacity)), polygon, 2)
        pygame.draw.line(panel, (122, 226, 239), (30, 0), (245, 0), 4)
        pygame.draw.line(panel, (122, 226, 239), (515, 269), (730, 269), 4)
        self.screen.blit(panel, rectangle)
        self.text('TACTICAL REPLAY  /  结算统计' if score is not None else 'TACTICAL REPLAY  /  对局记录', (365, 426), 19, (100, 194, 215))
        if score is None:
            self.fitted_text(title, (365, 482), 685, 44, bold=True)
        else:
            self.text(title, (365, 467), 26, (188, 210, 230))
            value = score * (1 - (1 - progress)**3)
            self.text(f'{value:05.2f}', (365, 504), 88, (255, 206, 140), True)
            if progress >= 1:
                pygame.draw.rect(self.screen, (65, 186, 159), (852, 508, 178, 57), 2, border_radius=8)
                self.text('记录完成', (877, 519), 26, (113, 223, 188), True)
        self.fitted_text(subtitle, (365, 638), 685, 21, (181, 199, 224))


def export_frame(data, output, simulation_seconds, font_file=None, latin_font_file=None, map_only=False):
    """导出真实时刻的无损PNG；静态画面不叠加片头、快进或结算遮罩。"""
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('静态帧输出应为.png文件')
    if output.exists() or output.with_suffix('.validation.json').exists():
        raise FileExistsError(output)
    meta, times, states, index, events, summary = data
    if (isinstance(simulation_seconds, bool) or not isinstance(simulation_seconds, (int, float))
            or not math.isfinite(simulation_seconds) or not 0 <= simulation_seconds <= times[-1]):
        raise ValueError('静态帧时刻须位于日志实际记录的时间范围内')
    simulation_step = recorded_step(meta, simulation_seconds)
    timeline = Timeline(times[-1], events, 112)
    painter = ReplayPainter(*data, timeline, font_file=font_file, latin_font_file=latin_font_file)
    video_t = timeline.video_time(simulation_seconds)
    image = painter.render(video_t, simulation_seconds=simulation_seconds, static=True)
    crop = pygame.Rect(24, 15, 1374, 924) if map_only else image.get_rect()
    image = image.subsurface(crop).copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(image, str(output))
    decoded = pygame.image.load(str(output))
    if (decoded.get_size() != image.get_size()
            or pygame.image.tobytes(decoded, 'RGB') != pygame.image.tobytes(image, 'RGB')):
        raise RuntimeError('静态帧PNG无损解码核验失败')
    destroyed = {event['target'] for event in events
                 if event['kind'] == 'objective_destroyed' and event['t'] <= simulation_seconds}
    objectives = {row['id']: row for row in summary['objectives']}
    score = 100 * sum(objectives[key]['weight'] for key in destroyed) / painter.total_weight
    receipt = {'image': str(output.resolve()), 'image_size': list(image.get_size()),
               'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
               'simulation_seconds': float(simulation_seconds), 'simulation_step': float(simulation_step),
               'clock_display': 'simulation_step', 'event_clock_display': 'simulation_step',
               'timeline_display': 'recorded_step_linear', 'score_at_frame': score,
               'map_only': bool(map_only), 'crop_pixels': list(crop),
               'coordinate_units': 'metres', 'axes': {'x':'east', 'y':'north'},
               'pixels_per_meter': painter.pixels_per_meter,
               'font_files': list(painter.font_files), 'source_audit': meta['audit'],
               'position_interpolation': 'linear_recorded_endpoints',
               'active_waypoint_guidance_count': len(painter.active_guidance(simulation_seconds,
                    states[min(max(bisect_right(times, simulation_seconds) - 1, 0), len(times) - 1)])),
               'waypoint_guidance': 'recorded_xy_dashed_amber_with_hollow_endpoint',
               'static_overlays': 'no_intro_outro_or_fast_forward', 'png_lossless_decode_passed': True}
    output.with_suffix('.validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def export(data, output, fps=30, duration=112, font_file=None, latin_font_file=None):
    args = argparse.Namespace(output=Path(output), fps=fps, duration=duration)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta, times, states, index, events, summary = data
    timeline = Timeline(times[-1], events, args.duration)
    painter = ReplayPainter(*data, timeline, font_file=font_file, latin_font_file=latin_font_file)
    frames = round(timeline.duration * args.fps)
    partial = args.output.with_suffix('.partial.mp4')
    encoder_log = args.output.with_suffix('.encoder.log')
    command = [str(FFMPEG), '-y', '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x1080', '-r', str(args.fps),
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
               '-profile:v', 'high', '-level:v', '4.2', '-pix_fmt', 'yuv420p', '-tag:v', 'avc1',
               '-threads', '2', '-movflags', '+faststart', str(partial)]
    with encoder_log.open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=subprocess.DEVNULL,
                                   creationflags=PROCESS_FLAGS)
        try:
            for frame in range(frames):
                image = painter.render(frame / args.fps)
                process.stdin.write(pygame.image.tobytes(image, 'RGB'))
                if frame % (args.fps * 10) == 0:
                    print(json.dumps({'frames': frame, 'total': frames}), flush=True)
        finally:
            process.stdin.close()
            exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError(encoder_log.read_text(encoding='utf-8'))
    # 全片严格解码，任何坏帧都会令交付失败。
    check = subprocess.run([str(FFMPEG), '-v', 'error', '-xerror', '-i', str(partial),
                            '-map', '0:v:0', '-vsync', '0', '-progress', 'pipe:1', '-f', 'null', '-'],
                           capture_output=True, text=True, encoding='utf-8', creationflags=PROCESS_FLAGS)
    if check.returncode:
        raise RuntimeError(check.stderr)
    decoded_frames = [int(line.split('=', 1)[1]) for line in check.stdout.splitlines() if line.startswith('frame=')][-1]
    if decoded_frames != frames:
        raise RuntimeError(f'视频帧数不符：{decoded_frames}，预期{frames}')
    probe = subprocess.run([str(FFMPEG), '-hide_banner', '-i', str(partial)], capture_output=True,
                           text=True, encoding='utf-8', creationflags=PROCESS_FLAGS)
    if '1920x1080' not in probe.stderr or 'yuv420p' not in probe.stderr or 'h264' not in probe.stderr:
        raise RuntimeError('视频规格核验失败：' + probe.stderr)
    partial.replace(args.output)
    if encoder_log.stat().st_size == 0:
        encoder_log.unlink()
    receipt = {'video': str(args.output.resolve()), 'size_bytes': args.output.stat().st_size,
               'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'fps': args.fps, 'frames': frames, 'duration_seconds': frames / args.fps,
               'resolution': [1920, 1080], 'codec': 'H.264 / yuv420p / faststart',
               'strict_full_decode_passed': True, 'decoded_frames': decoded_frames,
               'score_recomputed': summary['score'],
               'source_steps': summary['steps'], 'source_audit': meta['audit'],
               'events': dict(Counter(row['kind'] for row in events)),
               'visual_settings': {'theme': 'projection_slate', 'map_color': list(ReplayPainter.MAP_COLOR),
                                   'font_families': ['FangSong', 'Times New Roman'],
                                   'font_files': list(painter.font_files), 'font_scale': ReplayPainter.FONT_SCALE,
                                   'title': '对局回放', 'defensive_caption': False,
                                   'unit_legend': 'human_and_assignment_top_bug_with_swarm_bottom', 'clock': 'header_left_of_legend',
                                   'clock_position': list(painter.clock_position),
                                   'legend_panel': list(painter.legend_box),
                                   'score_panel': list(painter.score_box), 'statistics_panel': list(painter.stats_box),
                                   'scene_metadata': 'map_footer',
                                   'timeline_panel': list(painter.timeline_box),
                                   'timeline_marker_shapes': 'hexagon_pentagon_diamond_cross_and_discovery_dot',
                                   'timeline_marker_size': ReplayPainter.TIMELINE_MARKER_SIZE,
                                   'timeline_destroyed_marker_style': 'colored_outline_transparent_interior_colored_cross',
                                   'timeline_event_kinds': ['mother_destroyed', 'sentry_destroyed', 'dark_destroyed', 'dark_discovered'],
                                   'timeline_discovery_grouping': 'none_each_recorded_event_at_its_time',
                                   'timeline_marker_layout': 'single_row_at_recorded_time_no_connectors',
                                   'assignment_lines': 'recorded_targets_dashed_amber',
                                   'waypoint_guidance': 'recorded_xy_dashed_amber_with_hollow_endpoint',
                                   'hit_line_colors': {str(key): list(value) for key, value in ReplayPainter.HIT_COLORS.items()},
                                   'discovered_dark_hive_color': list(Icons.TYPE_COLORS[9500]),
                                   'secondary_text_color': list(ReplayPainter.SECONDARY_TEXT),
                                   'rubble': True, 'rubble_style': 'small_dark_gray_cross_only',
                                   'rubble_color': list(ReplayPainter.REMNANT_COLOR), 'rubble_scale': ReplayPainter.REMNANT_SCALE,
                                   'red_unit_scale': ReplayPainter.RED_SCALE, 'red_heading': 'shortest_arc_interpolation_1_degree_draw_cache',
                                   'hive_scale': ReplayPainter.HIVE_SCALE,
                                   'normal_playback_speed': timeline.normal_speed,
                                   'normal_playback_seconds_saved': timeline.normal_seconds_saved,
                                   'score_rule_caption': False,
                                   'red_halo': False, 'red_ring': '1px_hard_edge_25_percent_alpha',
                                   'red_trail_seconds': ReplayPainter.RED_TRAIL_SECONDS,
                                   'red_trail_alpha_max': ReplayPainter.RED_TRAIL_ALPHA / 255,
                                   'red_trail_shape': 'narrow_quadratic_taper_no_dots',
                                   'satellite_fill_alpha_max': ReplayPainter.SATELLITE_FILL_ALPHA / 255,
                                   'motion_trails': True, 'unit_labels': 'key_event_or_damage_only',
                                   'faction_terms': ['人族单位', '虫族单位'],
                                   'troops': 'mobile_units_alive_loss_from_recorded_hp_self_destruction_included',
                                   'event_feed_rows': ReplayPainter.EVENT_ROWS,
                                   'event_feed_slide_seconds': ReplayPainter.EVENT_SLIDE_SECONDS,
                                   'event_feed_order': 'newest_first',
                                   'event_feed_non_score_fold_seconds': ReplayPainter.EVENT_FOLD_SECONDS,
                                   'event_feed_clip': 'isolated_layer_and_fixed_six_slots',
                                   'position_interpolation': 'linear_recorded_endpoints',
                                   'minimum_legend_and_timeline_font_pixels': 21,
                                   'gain_capsules': True, 'gain_collision_spacing_pixels': 24,
                                   'fast_forward_notice': 'map_center_symmetric_breathing_triple_chevrons', 'score_particle': 'bezier_trail',
                                   'score_feedback': 'digits_and_card_glow',
                                   'finder_labels': False, 'high_red_caption': False, 'semantic_timeline': True,
                                   'discovery_tower_highlight': False, 'satellite_scene_effect': True,
                                   'score_updates_at_event_time': True, 'idle_tail_max_seconds': 3},
               'visual_inspection_performed': False,
               'mobile_troops_initial_alive_loss_deployed': painter.troop_statistics(0).tolist(),
               'mobile_troops_final_alive_loss_deployed': painter.troop_statistics(times[-1]).tolist(),
               'event_video_seconds': [{**event, 'video_t': timeline.video_time(event['t'])} for event in events],
               'ffmpeg_probe': probe.stderr}
    args.output.with_suffix('.validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt
