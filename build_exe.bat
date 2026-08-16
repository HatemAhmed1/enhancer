@echo off
REM Builds dist\Enhancer\Enhancer.exe
REM
REM Takes 10-20 minutes and produces roughly 5 GB, most of which is PyTorch
REM and the CUDA runtime. Nothing to be done about that: the graphics code is
REM the application.

cd /d "%~dp0"

if not exist ".venv\Scripts\pyinstaller.exe" (
    echo Installing PyInstaller...
    ".venv\Scripts\python.exe" -m pip install pyinstaller
)

echo Building. This takes a while.
echo.
".venv\Scripts\pyinstaller.exe" enhancer.spec --noconfirm
if errorlevel 1 (
    echo.
    echo Build failed. See the output above.
    pause
    exit /b 1
)

echo.
echo Built: dist\Enhancer\Enhancer.exe
echo.
echo Copy the whole dist\Enhancer folder to move it — the exe needs the files
echo beside it. Models are read from models\custom and models\rife next to the
echo executable.
pause
