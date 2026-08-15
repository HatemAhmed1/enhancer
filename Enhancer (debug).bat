@echo off
REM Launches the window with a visible console so startup errors are readable.
REM Use this when the normal desktop shortcut does nothing.
cd /d "%~dp0"
echo Starting Enhancer with console output...
echo.
".venv\Scripts\python.exe" -m enhancer.cli gui
echo.
echo Enhancer exited with code %errorlevel%.
pause
