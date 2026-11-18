import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from replay_tool.log_input import load, save_compact


def main():
    parser = argparse.ArgumentParser(description='读取米制坐标对局日志，导出回放视频或真实静态帧')
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('audit', 'render', 'compact', 'frame'):
        child = subparsers.add_parser(command)
        child.add_argument('log', type=Path, help='SDK逐行JSON日志，或紧凑回放目录')
        child.add_argument('--step-seconds', type=float, help='旧日志缺少时钟时，明确指定每仿真步秒数')
        if command in ('render', 'compact', 'frame'):
            child.add_argument('output', type=Path)
        if command == 'render':
            child.add_argument('--fps', type=int, default=30)
            child.add_argument('--duration', type=float, default=112)
        if command in ('render', 'frame'):
            child.add_argument('--font-file', type=Path)
            child.add_argument('--latin-font-file', type=Path)
        if command == 'frame':
            child.add_argument('--step', type=int, required=True, help='真实仿真步号')
            child.add_argument('--map-only', action='store_true', help='仅导出地图与图例，适合PPT')
    args = parser.parse_args()
    if args.command == 'render':
        if args.output.suffix.lower() != '.mp4' or args.output.exists():
            parser.error('输出应为尚不存在的.mp4文件')
        if not 15 <= args.duration <= 600:
            parser.error('输出时长须为15至600秒')
        if not 1 <= args.fps <= 60:
            parser.error('输出帧率须为1至60的整数')
    if args.command == 'frame':
        if args.output.suffix.lower() != '.png' or args.output.exists():
            parser.error('输出应为尚不存在的.png文件')
        if args.step < 0:
            parser.error('仿真步号须为非负整数')
    data = load(args.log, args.step_seconds)
    print(json.dumps(data[0]['audit'], ensure_ascii=False, indent=2), flush=True)
    if args.command == 'render':
        from replay_tool.video import export
        receipt = export(data, args.output, args.fps, args.duration,
                         str(args.font_file) if args.font_file else None,
                         str(args.latin_font_file) if args.latin_font_file else None)
        print(json.dumps({key: receipt[key] for key in ('video', 'duration_seconds', 'score_recomputed')}, ensure_ascii=False))
    elif args.command == 'compact':
        save_compact(data, args.output)
        print(str(args.output.resolve()))
    elif args.command == 'frame':
        from replay_tool.video import export_frame
        simulation_seconds = args.step * data[0]['step_seconds']
        if simulation_seconds > data[1][-1]:
            parser.error('所选步号超出日志范围')
        receipt = export_frame(
            data, args.output, simulation_seconds,
            str(args.font_file) if args.font_file else None,
            str(args.latin_font_file) if args.latin_font_file else None,
            map_only=args.map_only,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
