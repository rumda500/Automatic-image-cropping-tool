@echo off
setlocal

cd /d "%~dp0"

if not exist .venv (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Python venv作成に失敗しました。
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo [ERROR] 仮想環境の有効化に失敗しました。
  pause
  exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] pip更新に失敗しました。
  pause
  exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] 依存インストールに失敗しました。
  pause
  exit /b 1
)

if not exist "%CD%\birefnet_finetuned_toonout.pth" (
  echo [INFO] Checkpointが見つからないためダウンロードします。
  python fetch_checkpoint.py --output "%CD%\birefnet_finetuned_toonout.pth"
  if errorlevel 1 (
    echo [ERROR] Checkpointダウンロードに失敗しました。
    pause
    exit /b 1
  )
)

python app.py --checkpoint "%CD%\birefnet_finetuned_toonout.pth" --device auto --host 127.0.0.1 --port 7860
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] UI起動中にエラーが発生しました。code=%EXIT_CODE%
  pause
)

exit /b %EXIT_CODE%
