@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0terrain.ps1" %*
exit /b %errorlevel%
