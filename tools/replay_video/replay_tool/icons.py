import math
import pygame


class Icons:
    TYPE_COLORS = {21000: (255, 82, 101), 21001: (255, 155, 78), 21002: (244, 103, 142),
                   9400: (83, 213, 242), 9500: (95, 168, 255), 9600: (80, 221, 185),
                   24000: (95, 168, 255)}

    def polygon(self, surf, cx, cy, size, kind, sides, filled=False):
        points = [(cx + math.cos(-math.pi / 2 + k * math.tau / sides) * size * .32,
                   cy + math.sin(-math.pi / 2 + k * math.tau / sides) * size * .32)
                  for k in range(sides)]
        pygame.draw.polygon(surf, self.TYPE_COLORS[kind], points, 0 if filled else 2)
        pygame.draw.circle(surf, self.TYPE_COLORS[kind] if not filled else (30, 28, 43), (cx, cy), max(2, size // 12))

    def draw_destroyer(self, surf, cx, cy, size):
        points = [(cx, cy - size * .43), (cx + size * .31, cy + size * .28),
                  (cx, cy + size * .10), (cx - size * .31, cy + size * .28)]
        pygame.draw.polygon(surf, self.TYPE_COLORS[21000], points)
        pygame.draw.polygon(surf, (255, 226, 199), points, 1)
        pygame.draw.line(surf, (255, 246, 223), (cx, cy - size * .23), (cx, cy + size * .04), 2)

    def draw_assault(self, surf, cx, cy, size):
        points = [(cx + math.cos(-math.pi / 2 + k * math.tau / 3) * size * .32,
                   cy + math.sin(-math.pi / 2 + k * math.tau / 3) * size * .32)
                  for k in range(3)]
        pygame.draw.polygon(surf, self.TYPE_COLORS[21001], points)
        pygame.draw.polygon(surf, (255, 244, 214), points, 1)

    def draw_scavenger(self, surf, cx, cy, size):
        color, radius = self.TYPE_COLORS[21002], max(3, int(size * .31))
        pygame.draw.circle(surf, color, (cx, cy), radius)
        pygame.draw.circle(surf, (255, 233, 224), (cx, cy), radius, 1)
        for sign in (-1, 1):
            pygame.draw.line(surf, (91, 29, 54), (cx - radius // 2, cy - sign * radius // 2),
                             (cx + radius // 2, cy + sign * radius // 2), 1)

    def draw_hive(self, surf, cx, cy, size):
        self.polygon(surf, cx, cy, size, 9400, 6, True)

    def draw_sentry_hive(self, surf, cx, cy, size):
        self.polygon(surf, cx, cy, size, 9600, 5)

    def draw_dark_hive(self, surf, cx, cy, size):
        self.polygon(surf, cx, cy, size, 9500, 4, True)

    def draw_swarm(self, surf, cx, cy, size):
        for angle in (-math.pi / 2, math.pi / 6, math.pi * 5 / 6):
            point = (round(cx + math.cos(angle) * size * .22), round(cy + math.sin(angle) * size * .22))
            pygame.draw.circle(surf, self.TYPE_COLORS[24000], point, max(2, int(size * .17)), 1)

    def event_marker(self, surface, center, size, event, filled=True):
        x, y = center
        radius = size * .44
        if event['kind'] == 'satellite':
            color = (186, 144, 255)
            pygame.draw.ellipse(surface, color, (x - radius, y - radius * .65, radius * 2, radius * 1.3), 2)
            pygame.draw.circle(surface, color, center, max(2, round(size * .13)))
        elif event['kind'] == 'dark_discovered':
            color = self.TYPE_COLORS[9500]
            points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
            pygame.draw.polygon(surface, color, points, 2)
            pygame.draw.circle(surface, color, center, 2)
        elif event['kind'] == 'high_red_discovered':
            color = (255, 155, 78)
            points = [(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)]
            pygame.draw.polygon(surface, color, points, 2)
            pygame.draw.line(surface, color, (x, y - radius * .35), (x, y + radius * .25), 2)
            pygame.draw.circle(surface, color, (x, round(y + radius * .55)), 1)
        else:
            kind = event['type']
            color = self.TYPE_COLORS[kind]
            sides = {9400: 6, 9600: 5, 9500: 4}[kind]
            points = [(x + math.cos(-math.pi / 2 + k * math.tau / sides) * radius,
                       y + math.sin(-math.pi / 2 + k * math.tau / sides) * radius) for k in range(sides)]
            pygame.draw.polygon(surface, color, points, 0 if filled else 2)
            for sign in (-1, 1):
                pygame.draw.line(surface, (18, 35, 53) if filled else color, (x - radius * .34, y - sign * radius * .34),
                                 (x + radius * .34, y + sign * radius * .34), 2)
