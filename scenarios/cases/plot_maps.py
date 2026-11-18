"""统一重绘正式场景信息图，并更新图片校验值和展示编号对照。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, patheffects, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from matplotlib.transforms import Bbox


CATALOGUE = Path(__file__).resolve().parents[1]
FONT_SIZE = 10
FIGURE_WIDTH = 6
DPI = 300
MARKER_SIZE = 7
STYLES = {
    "target": ("母巢", "o", "#bb7d10"),
    "site": ("哨卫巢", "s", "#2869ab"),
    "ship": ("暗巢", "D", "#238865"),
}
PREFIXES = {"target": "M", "site": "S", "ship": "A"}
ROOT_KINDS = {9400: "target", 9600: "site", 9500: "ship"}
# 与原生哨卫探测模型一致，静态图展示 20 公里的水平投影。
DETECTION_RANGE_KM = 20


def configure_font(path: Path | None, latin_path: Path | None = None) -> None:
    """英文和数字采用 Times New Roman，中文采用宋体，字号统一。"""
    if latin_path is not None:
        font_manager.fontManager.addfont(latin_path)
        latin_family = font_manager.FontProperties(fname=latin_path).get_name()
    else:
        latin_family = "Times New Roman"
        font_manager.findfont(latin_family, fallback_to_default=False)
    if path is not None:
        font_manager.fontManager.addfont(path)
        chinese_family = font_manager.FontProperties(fname=path).get_name()
    else:
        available = set(font_manager.fontManager.get_font_names())
        chinese_family = next((name for name in (
            "SimSun", "NSimSun", "宋体", "FandolSong",
        ) if name in available), None)
        if chinese_family is None:
            raise ValueError("请使用 --font 指定宋体字体文件。")
    plt.rcParams.update({
        "font.family": [latin_family, chinese_family],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def place_labels(ax, points: list[tuple[float, float]], display_ids: list[str]) -> None:
    """在屏幕坐标中挑选标注方向，避开图例、标记、已有文字和边界。"""
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    pixels_per_point = figure.dpi / 72
    centers = [ax.transData.transform(point) for point in points]
    radius = (MARKER_SIZE / 2 + 1) * pixels_per_point
    markers = [Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)
               for x, y in centers]
    labels = [ax.get_legend().get_window_extent(renderer).padded(2 * pixels_per_point)]
    font = font_manager.FontProperties(size=FONT_SIZE)
    directions = ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
    for text, point, center in zip(display_ids, points, centers):
        width, height, descent = renderer.get_text_width_height_descent(text, font, False)
        height += descent + 2 * pixels_per_point
        choices = []
        for offset in (6, 10, 14, 18):
            for dx, dy in directions:
                x = center[0] + dx * offset * pixels_per_point
                y = center[1] + dy * offset * pixels_per_point
                left = x if dx > 0 else x - width if dx < 0 else x - width / 2
                bottom = y if dy > 0 else y - height if dy < 0 else y - height / 2
                box = Bbox.from_bounds(left, bottom, width, height).padded(pixels_per_point)
                outside = not (ax.bbox.contains(box.x0, box.y0)
                               and ax.bbox.contains(box.x1, box.y1))
                overlaps = sum(box.overlaps(other) for other in markers + labels)
                choices.append((outside * 10000 + overlaps * 1000 + offset,
                                dx, dy, offset, box))
        _, dx, dy, offset, box = min(choices, key=lambda item: item[0])
        labels.append(box)
        ax.annotate(text, point, xytext=(dx * offset, dy * offset),
                    textcoords="offset points", fontsize=FONT_SIZE,
                    ha="left" if dx > 0 else "right" if dx < 0 else "center",
                    va="bottom" if dy > 0 else "top" if dy < 0 else "center",
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                    zorder=4)


def map_size_km(area: dict) -> tuple[float, float]:
    """读取场景平面宽高，并把米转换为公里。"""
    return (area["xMax"] - area["xMin"]) / 1000, (area["yMax"] - area["yMin"]) / 1000


def draw_case(scenario: dict, output: Path) -> list[dict]:
    """以场景地图中心为原点，按公里等比例绘制初始布局。"""
    profile = scenario["imagineProfile"]
    metadata = scenario["metadata"]
    area = profile["mapArea"]
    width, height = map_size_km(area)

    def position(x, y):
        return (x - (area["xMin"] + area["xMax"]) / 2) / 1000, (y - (area["yMin"] + area["yMax"]) / 2) / 1000

    # 用物理英寸控制留白，保持整张图片宽六英寸和横纵坐标等比例。
    left, right, bottom, top = 0.58, 0.08, 0.44, 0.28
    axes_width = FIGURE_WIDTH - left - right
    axes_height = axes_width * height / width
    figure_height = axes_height + bottom + top
    figure = plt.figure(figsize=(FIGURE_WIDTH, figure_height), dpi=DPI, facecolor="white")
    ax = figure.add_axes((left / FIGURE_WIDTH, bottom / figure_height,
                          axes_width / FIGURE_WIDTH, axes_height / figure_height))
    ax.set(xlim=(-width / 2, width / 2), ylim=(-height / 2, height / 2),
           xlabel="X (km)", ylabel="Y (km)",
           title=f"{metadata['id']}：{metadata['title']}", aspect="equal")
    ax.set_title(ax.get_title(), pad=5)
    ax.tick_params(pad=2, length=3)
    ax.grid(color="#dedede", linewidth=0.4, alpha=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#999999", linewidth=0.6, zorder=0.6)
    ax.axvline(0, color="#999999", linewidth=0.6, zorder=0.6)

    # 人族部署区直接读取 redArea，用斜线纹路和居中文字说明。
    for polygon in profile["redArea"]["coordinates"]:
        vertices = [position(x, y) for x, y in polygon]
        ax.add_patch(Polygon(vertices, closed=True, facecolor="#fdf5f5", edgecolor="#c98181",
                             linewidth=0.8, hatch="//", zorder=0.2))
        center = tuple(sum(point[axis] for point in vertices) / len(vertices) for axis in (0, 1))
        ax.text(*center, "人族部署区域", ha="center", va="center", color="#954848",
                bbox={"facecolor": "#fdf5f5", "edgecolor": "none", "pad": 3}, zorder=4)

    # 探测范围沿用原图的水平投影表达，先画范围，再画实体及编号。
    roots = [entry["entity"] for entry in profile["entityList"]
             if entry["entity"]["sideId"] == 1 and entry["entity"]["parentId"] == -1
             and entry["entity"]["entityType"] in ROOT_KINDS]
    roots.sort(key=lambda root: (tuple(STYLES).index(ROOT_KINDS[root["entityType"]]), root["id"]))
    kinds = [ROOT_KINDS[root["entityType"]] for root in roots]
    points = [position(root["pos"]["x"], root["pos"]["y"]) for root in roots]
    # 每张地图中，母巢、哨卫巢、暗巢各自从 M00、S00、A00 开始编号。
    counts = dict.fromkeys(STYLES, 0)
    display_ids = []
    for kind in kinds:
        display_ids.append(f"{PREFIXES[kind]}{counts[kind]:02d}")
        counts[kind] += 1
    for kind, point in zip(kinds, points):
        if kind == "site":
            ax.add_patch(Circle(point, DETECTION_RANGE_KM, facecolor="#77acd2", edgecolor="#77acd2",
                                alpha=0.13, linewidth=0.6, zorder=1))
    for kind, (_, marker, color) in STYLES.items():
        selected = [point for root_kind, point in zip(kinds, points) if root_kind == kind]
        if selected:
            ax.scatter(*zip(*selected), s=MARKER_SIZE ** 2, marker=marker, c=color,
                       edgecolors="white", linewidths=0.4, zorder=3)

    # 图例位于图内左下角，使用白色不透明背景，保留场景原始坐标范围。
    handles = [Line2D([], [], linestyle="none", marker=marker, markersize=MARKER_SIZE,
                      markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.4,
                      label=label) for label, marker, color in STYLES.values()]
    ax.legend(handles=handles, loc="lower left",
              borderaxespad=0.5, framealpha=1.0, facecolor="white", edgecolor="#cccccc",
              handletextpad=0.4, labelspacing=0.4)
    place_labels(ax, points, display_ids)
    figure.savefig(output, dpi=DPI, facecolor="white")
    plt.close(figure)
    return [{"scenario_id": metadata["id"], "display_id": display_id,
             "entity_id": root["id"], "kind": kind}
            for display_id, root, kind in zip(display_ids, roots, kinds)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="宋体字体文件，例如 simsun.ttc 或 FandolSong-Regular.otf")
    parser.add_argument("--latin-font", type=Path, help="Times New Roman 字体文件，例如 times.ttf")
    args = parser.parse_args()
    configure_font(args.font, args.latin_font)
    manifest_path = CATALOGUE / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    image_directory = CATALOGUE / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    mapping = []
    for case in manifest["cases"]:
        scenario_path = CATALOGUE / case["scenario"]
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        output = image_directory / f"{scenario_path.stem}.png"
        mapping.extend(draw_case(scenario, output))
        image_key = output.relative_to(CATALOGUE).as_posix()
        case["files_sha256"][case["scenario"]] = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
        case["files_sha256"][image_key] = hashlib.sha256(output.read_bytes()).hexdigest()
        print(f"已重绘 {case['id']}：{output.relative_to(CATALOGUE)}")

    # 集中保留展示编号与原始实体 ID 的对应关系，场景仍使用原始 ID。
    with (CATALOGUE / "map_labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("scenario_id", "display_id", "entity_id", "kind"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(mapping)
    newline = "\r\n" if b"\r\n" in original else "\n"
    manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
                              .replace("\n", newline).encode("utf-8"))


if __name__ == "__main__":
    main()
