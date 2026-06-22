@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0.."
echo.
echo ========================================
echo   A股量化回测系统 - Web版
echo ========================================
echo.
echo 正在检查依赖...
call .venv\Scripts\activate.bat
pip install flask -q 2>nul
cd webapp
echo.
echo 启动服务...
echo 浏览器将自动打开 http://localhost:5000
echo 按 Ctrl+C 停止服务
echo.
python app.py
pause
