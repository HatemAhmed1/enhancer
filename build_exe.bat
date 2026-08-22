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

REM Models are read from beside the executable, and PyInstaller does not put
REM them there. Without this the fresh build starts, finds nothing to enlarge
REM with, and has to be pointed at a folder by hand.
echo.
echo Copying models beside the executable...
if not exist "dist\Enhancer\models\custom" mkdir "dist\Enhancer\models\custom"
if not exist "dist\Enhancer\models\rife" mkdir "dist\Enhancer\models\rife"
if exist "models\custom\*.pth" copy /y "models\custom\*.pth" "dist\Enhancer\models\custom\" >nul
if exist "models\rife\*.pkl" copy /y "models\rife\*.pkl" "dist\Enhancer\models\rife\" >nul

echo.
echo Built: dist\Enhancer\Enhancer.exe
echo.
echo Checking it can run...
"dist\Enhancer\Enhancer.exe" check
echo.
echo Copy the whole dist\Enhancer folder to move it — the exe needs the files
echo beside it, models included.
pause
