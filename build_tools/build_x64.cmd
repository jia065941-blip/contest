@echo off
rem ===========================================================================
rem  CPython 3.11 x64 串行编译，再组装完整独立运行目录
rem
rem  前置条件（缺一不可）:
rem    1. python 指向 CPython 3.11 x64，已安装 build_tools\requirements-build.txt
rem    2. 已安装 "Desktop development with C++"；MSVC 由 setuptools 自动定位
rem
rem  用法:
rem      build_tools\build_x64.cmd
rem ===========================================================================

chcp 65001 >nul
cd /d "%~dp0.."

echo [1/2] 开始编译: .py -^> .pyd
python build_tools\cython_setup.py build_ext
if errorlevel 1 (
    echo [ERROR] 编译失败，请查看上方日志
    exit /b 1
)

echo [2/2] 编译完成，开始组装和核验完整运行目录
python build_tools\finalize_dist.py
if errorlevel 1 (
    echo [ERROR] 保留文件复制失败，请查看上方日志
    exit /b 1
)

echo 全部完成，产物位于 build\envengine_dist
