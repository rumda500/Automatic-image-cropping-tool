@echo off
setlocal

cd /d "%~dp0"

:menu
echo.
echo =========== BiRefNet-WebUI Launcher ===========
echo   1^) Web UIを起動
echo   2^) CLIヘルプを表示
echo   q^) 終了
echo ===============================================
echo.
set /p CHOICE=選択してください: 

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
  echo 終了します。
  exit /b 0
)

echo [WARN] 無効な入力です: %CHOICE%
goto menu
