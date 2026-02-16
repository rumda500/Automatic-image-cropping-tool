@echo off
setlocal

cd /d "%~dp0"

:menu
echo.
echo =========== BiRefNet-WebUI Launcher ===========
echo   1^) Start Web UI
echo   2^) Show CLI help
echo   q^) Quit
echo ===============================================
echo.
set /p CHOICE=Select an option: 

if /i "%CHOICE%"=="1" (
  call run_ui.bat
  goto menu
)

if /i "%CHOICE%"=="2" (
  if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe cutout_tool.py --help
  ) else (
    py -3 cutout_tool.py --help
  )
  goto menu
)

if /i "%CHOICE%"=="q" (
  echo Exit.
  exit /b 0
)

echo [WARN] Invalid option: %CHOICE%
goto menu
