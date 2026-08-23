@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_local.ps1" %*
exit /b %ERRORLEVEL%
