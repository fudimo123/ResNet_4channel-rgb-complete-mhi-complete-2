@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "APP_FILE=%SCRIPT_DIR%app.py"
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "PYTHON_EXE="

if exist "%VENV_PYTHON%" (
    call :check_python "%VENV_PYTHON%"
    if defined PYTHON_EXE (
        goto python_selected
    ) else (
        echo .venv exists but is missing required packages.
    )
)

if not defined PYTHON_EXE (
    call :check_python python
)

if not defined PYTHON_EXE (
    call :check_py_launcher
)

:python_selected
if not defined PYTHON_EXE (
    echo.
    echo No usable Python environment was found.
    echo Required packages: flask, torch, torchvision, cv2, ultralytics, pillow
    echo.
    echo You can install them with one of these commands:
    echo   .venv\Scripts\python.exe -m pip install flask ultralytics opencv-python pillow
    echo   python -m pip install flask ultralytics opencv-python pillow
    echo.
    pause
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

echo ===============================================
echo Starting realtime detection UI...
echo App file: %APP_FILE%
echo Python: %PYTHON_EXE%
echo Browser URL: http://127.0.0.1:5000
echo ===============================================

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 6; Start-Process 'http://127.0.0.1:5000/'"
if /i "%PYTHON_EXE%"=="py -3" (
    py -3 "%APP_FILE%"
) else (
    "%PYTHON_EXE%" "%APP_FILE%"
)

echo.
echo The realtime detection server has stopped.
pause
exit /b 0

:check_python
"%~1" -c "import flask, torch, torchvision, cv2, ultralytics, PIL" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=%~1"
)
exit /b 0

:check_py_launcher
py -3 -c "import flask, torch, torchvision, cv2, ultralytics, PIL" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=py -3"
)
exit /b 0
