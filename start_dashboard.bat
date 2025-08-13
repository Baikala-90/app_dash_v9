@echo off
REM ====== Bookk Dashboard Autostart ======
setlocal ENABLEDELAYEDEXPANSION

REM --- Settings ---
set "PROJECT_DIR=C:\Users\BOOKK_PRINT\발주량_대시보드"
set "APP_FILE=app_dash_v6.py"
set "LOG_DIR=%PROJECT_DIR%\logs"

REM Optional: open to LAN and fix port
set "DASH_HOST=0.0.0.0"
set "DASH_PORT=8090"

REM --- Prepare ---
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
cd /d "%PROJECT_DIR%"

REM Give network and Google auth some time after boot
timeout /t 15 /nobreak >nul

REM --- Choose Python ---
set "PY_EXE="
if exist "%PROJECT_DIR%\venv\Scripts\activate.bat" (
  call "%PROJECT_DIR%\venv\Scripts\activate.bat"
  set "PY_EXE=python"
) else (
  if exist "C:\Users\BOOKK_PRINT\AppData\Local\Programs\Python\Python313\python.exe" (
    set "PY_EXE=C:\Users\BOOKK_PRINT\AppData\Local\Programs\Python\Python313\python.exe"
  ) else (
    for %%I in (python.exe) do set "PY_EXE=%%~$PATH:I"
  )
)

if not defined PY_EXE (
  echo [%date% %time%] ERROR: Python not found. >> "%LOG_DIR%\server.log"
  exit /b 1
)

echo [%date% %time%] Starting %APP_FILE% using %PY_EXE% >> "%LOG_DIR%\server.log"
"%PY_EXE%" "%PROJECT_DIR%\%APP_FILE%" >> "%LOG_DIR%\server.log" 2>&1
