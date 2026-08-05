#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


def calc_file_lines_size(filename):
    """简单版本：读取文件并计算每行大小"""
    filepath = Path(filename)

    if not filepath.exists():
        print(f"文件不存在: {filename}")
        return

    print(f"{'行号':<10} {'大小(字节)':<15} {'累积大小':<15}")
    print("-" * 50)

    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            # 计算字节数（UTF-8编码）
            size = len(line.encode('utf-8'))
            total += size
            print(f"{i:<10} {size:<15,} {total:<15,}")


if __name__ == "__main__":
    calc_file_lines_size("G:\\PycharmProjects\\competition-platform-env\\results\\20260703135440")
