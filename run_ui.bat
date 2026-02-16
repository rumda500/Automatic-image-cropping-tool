@echo off
setlocal

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.
  pause
  exit /b 1
)

"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install requirements.
  pause
  exit /b 1
)

if not exist "%CD%\birefnet_finetuned_toonout.pth" (
  echo [INFO] Checkpoint not found. Downloading...
  "%VENV_PY%" fetch_checkpoint.py --output "%CD%\birefnet_finetuned_toonout.pth"
  if errorlevel 1 (
    echo [ERROR] Failed to download checkpoint.
    pause
    exit /b 1
  )
)

"%VENV_PY%" app.py --checkpoint "%CD%\birefnet_finetuned_toonout.pth" --device auto --host 127.0.0.1 --port 7860
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] UI exited with error code %EXIT_CODE%.
  pause
)

exit /b %EXIT_CODE%
