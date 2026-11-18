from pathlib import Path

import pygame


class ReplayFont:
    def __init__(self, chinese_file, latin_file, size, bold=False):
        self.chinese = pygame.font.Font(chinese_file, size)
        self.latin = pygame.font.Font(latin_file, size)
        self.chinese.set_bold(bold)
        self.latin.set_bold(bold)
        self.baseline = max(font.get_ascent() for font in (self.chinese, self.latin))
        self.height = self.baseline + max(font.get_height() - font.get_ascent()
                                         for font in (self.chinese, self.latin))

    def runs(self, value):
        start = 0
        while start < len(value):
            latin = ord(value[start]) < 0x0250
            end = start + 1
            while end < len(value) and (ord(value[end]) < 0x0250) == latin:
                end += 1
            yield self.latin if latin else self.chinese, value[start:end]
            start = end

    def size(self, value):
        return sum(font.size(run)[0] for font, run in self.runs(value)), self.height

    def get_height(self):
        return self.height

    def render(self, value, antialias, color):
        image = pygame.Surface((max(1, self.size(value)[0]), self.height), pygame.SRCALPHA)
        x = 0
        for font, run in self.runs(value):
            glyph = font.render(run, antialias, color)
            image.blit(glyph, (x, self.baseline - font.get_ascent()))
            x += glyph.get_width()
        return image


def font_files(chinese_file=None, latin_file=None):
    chinese = chinese_file or pygame.font.match_font('fangsong')
    latin = latin_file or pygame.font.match_font('timesnewroman')
    if not chinese or not Path(chinese).is_file():
        raise ValueError('缺少仿宋字体，请安装仿宋或用--font-file指定字体文件')
    if not latin or not Path(latin).is_file():
        raise ValueError('缺少Times New Roman字体，请安装该字体或用--latin-font-file指定字体文件')
    return str(chinese), str(latin)
