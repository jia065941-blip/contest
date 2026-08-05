@echo off
set "PYTHONPATH=C:\tmp\competition_platform_runtime"
set "PYTHONDONTWRITEBYTECODE=1"
"C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe" "%~dp0run.py" %*
