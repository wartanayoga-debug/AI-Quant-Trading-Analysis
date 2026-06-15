@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM QuantPrime / AI SignalPro launcher
REM - Starts the Node app reliably on http://127.0.0.1:3000
REM - Starts Python bridge when available, but does not block the app
REM - Avoids heavy Python installs unless INSTALL_BRIDGE_DEPS=1 is set
REM ============================================================

title QuantPrime Launcher
cd /d "%~dp0"

echo.
echo ============================================================
echo  QuantPrime / AI SignalPro Launcher
echo ============================================================
echo  Project folder: %CD%
echo.

if not exist "package.json" (
  echo [ERROR] package.json not found.
  echo Put this .bat file in the project root folder.
  pause
  exit /b 1
)

if not exist "server.ts" (
  echo [ERROR] server.ts not found.
  echo This launcher must run from the AI SignalPro project root.
  pause
  exit /b 1
)

if not exist "logs" mkdir "logs"
if not exist "data" mkdir "data"

REM -------------------------------
REM Runtime defaults
REM -------------------------------
if "%PORT%"=="" set PORT=3000
if "%HOST%"=="" set HOST=127.0.0.1
if "%BRIDGE_PORT%"=="" set BRIDGE_PORT=8765
if "%BRIDGE_URL%"=="" set BRIDGE_URL=http://127.0.0.1:%BRIDGE_PORT%
if "%START_BRIDGE%"=="" set START_BRIDGE=1
if "%INSTALL_BRIDGE_DEPS%"=="" set INSTALL_BRIDGE_DEPS=0
if "%OPEN_BROWSER%"=="" set OPEN_BROWSER=1
if "%DRY_RUN%"=="" set DRY_RUN=0
if "%VENV_DIR%"=="" set VENV_DIR=.venv-ag

REM Load bridge model settings from .env before applying launcher fallbacks.
REM Existing shell variables still win, so advanced users can override per run.
if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
    set "ENV_KEY=%%A"
    set "ENV_VAL=%%B"
    if /I "!ENV_KEY!"=="ENABLE_CHRONOS" if "!ENABLE_CHRONOS!"=="" set "ENABLE_CHRONOS=!ENV_VAL!"
    if /I "!ENV_KEY!"=="CHRONOS_BACKEND" if "!CHRONOS_BACKEND!"=="" set "CHRONOS_BACKEND=!ENV_VAL!"
    if /I "!ENV_KEY!"=="CHRONOS_MODEL_ID" if "!CHRONOS_MODEL_ID!"=="" set "CHRONOS_MODEL_ID=!ENV_VAL!"
    if /I "!ENV_KEY!"=="CHRONOS_FALLBACK_MODEL_ID" if "!CHRONOS_FALLBACK_MODEL_ID!"=="" set "CHRONOS_FALLBACK_MODEL_ID=!ENV_VAL!"
    if /I "!ENV_KEY!"=="CHRONOS_DEVICE" if "!CHRONOS_DEVICE!"=="" set "CHRONOS_DEVICE=!ENV_VAL!"
    if /I "!ENV_KEY!"=="ENABLE_KRONOS" if "!ENABLE_KRONOS!"=="" set "ENABLE_KRONOS=!ENV_VAL!"
    if /I "!ENV_KEY!"=="KRONOS_MODEL_ID" if "!KRONOS_MODEL_ID!"=="" set "KRONOS_MODEL_ID=!ENV_VAL!"
    if /I "!ENV_KEY!"=="KRONOS_TOKENIZER_ID" if "!KRONOS_TOKENIZER_ID!"=="" set "KRONOS_TOKENIZER_ID=!ENV_VAL!"
    if /I "!ENV_KEY!"=="KRONOS_MAX_CONTEXT" if "!KRONOS_MAX_CONTEXT!"=="" set "KRONOS_MAX_CONTEXT=!ENV_VAL!"
    if /I "!ENV_KEY!"=="KRONOS_DEVICE" if "!KRONOS_DEVICE!"=="" set "KRONOS_DEVICE=!ENV_VAL!"
    if /I "!ENV_KEY!"=="ENABLE_FINBERT" if "!ENABLE_FINBERT!"=="" set "ENABLE_FINBERT=!ENV_VAL!"
    if /I "!ENV_KEY!"=="FINBERT_MODEL_ID" if "!FINBERT_MODEL_ID!"=="" set "FINBERT_MODEL_ID=!ENV_VAL!"
    if /I "!ENV_KEY!"=="FINBERT_DEVICE" if "!FINBERT_DEVICE!"=="" set "FINBERT_DEVICE=!ENV_VAL!"
  )
)

if "%ENABLE_CHRONOS%"=="" set ENABLE_CHRONOS=true
if "%CHRONOS_BACKEND%"=="" set CHRONOS_BACKEND=auto
if "%CHRONOS_MODEL_ID%"=="" set CHRONOS_MODEL_ID=amazon/chronos-2
if "%CHRONOS_FALLBACK_MODEL_ID%"=="" set CHRONOS_FALLBACK_MODEL_ID=amazon/chronos-t5-small
if "%CHRONOS_DEVICE%"=="" set CHRONOS_DEVICE=cpu
if "%ENABLE_KRONOS%"=="" set ENABLE_KRONOS=true
if "%KRONOS_MODEL_ID%"=="" set KRONOS_MODEL_ID=NeoQuasar/Kronos-small
if "%KRONOS_TOKENIZER_ID%"=="" set KRONOS_TOKENIZER_ID=NeoQuasar/Kronos-Tokenizer-base
if "%KRONOS_MAX_CONTEXT%"=="" set KRONOS_MAX_CONTEXT=512
if "%KRONOS_DEVICE%"=="" set KRONOS_DEVICE=cpu
if "%ENABLE_FINBERT%"=="" set ENABLE_FINBERT=true
if "%FINBERT_MODEL_ID%"=="" set FINBERT_MODEL_ID=ProsusAI/finbert
if "%FINBERT_DEVICE%"=="" set FINBERT_DEVICE=cpu
if "%ENABLE_BRIDGE_IN_SCAN%"=="" set ENABLE_BRIDGE_IN_SCAN=false
if "%ENABLE_SYNTHETIC_ENSEMBLE%"=="" set ENABLE_SYNTHETIC_ENSEMBLE=false

if "%LM_STUDIO_BASE_URL%"=="" set LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
if "%LM_STUDIO_API_KEY%"=="" set LM_STUDIO_API_KEY=lm-studio
if "%LM_STUDIO_MODEL%"=="" set LM_STUDIO_MODEL=local-model
if "%LM_STUDIO_TIMEOUT_MS%"=="" set LM_STUDIO_TIMEOUT_MS=12000

echo [INFO] App URL      : http://%HOST%:%PORT%
echo [INFO] Bridge URL   : %BRIDGE_URL%
echo [INFO] Start bridge : %START_BRIDGE%
echo [INFO] Install bridge deps: %INSTALL_BRIDGE_DEPS%
echo [INFO] Dry run      : %DRY_RUN%
echo [INFO] Python venv  : %VENV_DIR%
echo.

REM -------------------------------
REM Node.js checks and install
REM -------------------------------
where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] node.exe was not found.
  echo Install Node.js 20 LTS or newer, then run this launcher again.
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm was not found.
  echo Reinstall Node.js with npm enabled.
  pause
  exit /b 1
)

echo [OK] Node:
node --version
echo [OK] npm:
call npm --version

if not exist "node_modules" (
  echo.
  echo [SETUP] Installing Node dependencies...
  call npm install
  if errorlevel 1 (
    echo [ERROR] npm install failed. See the terminal output above.
    pause
    exit /b 1
  )
) else (
  echo [OK] node_modules already exists.
)

REM -------------------------------
REM Pick Python for optional bridge
REM -------------------------------
set PY_CMD=
set PY_TAG=

where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 --version >nul 2>nul
  if not errorlevel 1 (
    set PY_CMD=py -3.11
    set PY_TAG=py -3.11
  ) else (
    py -3 --version >nul 2>nul
    if not errorlevel 1 (
      set PY_CMD=py -3
      set PY_TAG=py -3
    )
  )
)

if "%PY_CMD%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 (
    set PY_CMD=python
    set PY_TAG=python
  )
)

if "%PY_CMD%"=="" (
  echo [WARN] Python was not found. Python bridge will be skipped.
  set START_BRIDGE=0
) else (
  echo [OK] Python selected: %PY_TAG%
  %PY_CMD% --version
)

REM -------------------------------
REM Optional Python bridge setup
REM -------------------------------
if "%START_BRIDGE%"=="1" (
  if not exist "python_bridge\bridge_server.py" (
    echo [WARN] python_bridge\bridge_server.py not found. Python bridge will be skipped.
    set START_BRIDGE=0
  )
)

if "%START_BRIDGE%"=="1" (
  if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [SETUP] Creating Python virtual environment in %VENV_DIR%...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
      echo [WARN] Failed to create %VENV_DIR%. Python bridge will be skipped.
      set START_BRIDGE=0
    )
  )
)

if "%START_BRIDGE%"=="1" (
  set VENV_PY=%CD%\%VENV_DIR%\Scripts\python.exe
  if "%INSTALL_BRIDGE_DEPS%"=="1" (
    echo.
    echo [SETUP] Installing Python bridge dependencies. This can take a while...
    "!VENV_PY!" -m pip install --upgrade pip wheel "setuptools<82"
    if errorlevel 1 goto bridge_deps_failed
    if exist "python_bridge\requirements.txt" (
      "!VENV_PY!" -m pip install -r "python_bridge\requirements.txt"
      if errorlevel 1 goto bridge_deps_failed
    )
    "!VENV_PY!" -m pip install --upgrade "setuptools<82"
    if errorlevel 1 goto bridge_deps_failed
    echo [OK] Python bridge dependencies installed.
  ) else (
    echo [INFO] Skipping Python dependency install.
    echo        To install/update bridge dependencies, run:
    echo        set INSTALL_BRIDGE_DEPS=1 ^&^& start_quantprime.bat
  )
)

if "%START_BRIDGE%"=="1" (
  "%CD%\%VENV_DIR%\Scripts\python.exe" -c "import fastapi, uvicorn" >nul 2>nul
  if errorlevel 1 (
    echo [WARN] Python bridge dependencies are not installed in %VENV_DIR%.
    echo        Bridge will be skipped so the app can start normally.
    echo        To install bridge dependencies, run:
    echo        set INSTALL_BRIDGE_DEPS=1 ^&^& start_quantprime.bat
    set START_BRIDGE=0
  )
)

goto after_bridge_deps

:bridge_deps_failed
echo [WARN] Python dependency setup failed. The Node app will still start with fallback models.
set START_BRIDGE=0

:after_bridge_deps

if "%DRY_RUN%"=="1" (
  echo.
  echo [OK] Dry run completed. Launcher checks passed; no app windows were started.
  endlocal
  exit /b 0
)

REM -------------------------------
REM Start Python bridge if possible
REM -------------------------------
if "%START_BRIDGE%"=="1" (
  echo.
  echo [START] Starting Python Bridge on port %BRIDGE_PORT%...
  start "QuantPrime Python Bridge" cmd /k ^
  "cd /d ""%CD%"" && ^
  set PORT=%BRIDGE_PORT%&& ^
  set BRIDGE_PORT=%BRIDGE_PORT%&& ^
  set ENABLE_CHRONOS=%ENABLE_CHRONOS%&& ^
  set CHRONOS_BACKEND=%CHRONOS_BACKEND%&& ^
  set CHRONOS_MODEL_ID=%CHRONOS_MODEL_ID%&& ^
  set CHRONOS_FALLBACK_MODEL_ID=%CHRONOS_FALLBACK_MODEL_ID%&& ^
  set CHRONOS_DEVICE=%CHRONOS_DEVICE%&& ^
  set ENABLE_KRONOS=%ENABLE_KRONOS%&& ^
  set KRONOS_MODEL_ID=%KRONOS_MODEL_ID%&& ^
  set KRONOS_TOKENIZER_ID=%KRONOS_TOKENIZER_ID%&& ^
  set KRONOS_MAX_CONTEXT=%KRONOS_MAX_CONTEXT%&& ^
  set KRONOS_DEVICE=%KRONOS_DEVICE%&& ^
  set ENABLE_FINBERT=%ENABLE_FINBERT%&& ^
  set FINBERT_MODEL_ID=%FINBERT_MODEL_ID%&& ^
  set FINBERT_DEVICE=%FINBERT_DEVICE%&& ^
  set ENABLE_SYNTHETIC_ENSEMBLE=%ENABLE_SYNTHETIC_ENSEMBLE%&& ^
  ""%CD%\%VENV_DIR%\Scripts\python.exe"" python_bridge\bridge_server.py"
  timeout /t 3 /nobreak >nul
) else (
  echo [INFO] Python bridge skipped. App will use backend fallback models where needed.
)

REM -------------------------------
REM Start Node app
REM -------------------------------
echo.
echo [START] Starting QuantPrime App...
start "QuantPrime App" cmd /k ^
"cd /d ""%CD%"" && ^
set HOST=%HOST%&& ^
set PORT=%PORT%&& ^
set BRIDGE_URL=%BRIDGE_URL%&& ^
set LM_STUDIO_BASE_URL=%LM_STUDIO_BASE_URL%&& ^
set LM_STUDIO_API_KEY=%LM_STUDIO_API_KEY%&& ^
set LM_STUDIO_MODEL=%LM_STUDIO_MODEL%&& ^
set LM_STUDIO_TIMEOUT_MS=%LM_STUDIO_TIMEOUT_MS%&& ^
set ENABLE_BRIDGE_IN_SCAN=%ENABLE_BRIDGE_IN_SCAN%&& ^
call npm run dev"

if "%OPEN_BROWSER%"=="1" (
  timeout /t 4 /nobreak >nul
  start "" "http://%HOST%:%PORT%"
)

echo.
echo ============================================================
echo Started.
echo App URL      : http://%HOST%:%PORT%
echo Bridge health: %BRIDGE_URL%/health
echo.
echo Notes:
echo - If Python bridge is skipped or offline, the app still runs.
echo - For full bridge dependencies, run:
echo   set INSTALL_BRIDGE_DEPS=1 ^&^& start_quantprime.bat
echo - To skip bridge:
echo   set START_BRIDGE=0 ^&^& start_quantprime.bat
echo ============================================================
echo.
pause
endlocal
