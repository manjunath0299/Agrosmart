@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo AgroSmart - Windows Setup
echo ==========================================

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python was not found.
  echo Install Python 3.12.x and run this again.
  pause
  exit /b 1
)

echo Creating .venv...
python -m venv .venv
if errorlevel 1 (
  echo ERROR: Could not create .venv
  pause
  exit /b 1
)

echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo Installing Python dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: Dependency installation failed.
  pause
  exit /b 1
)

echo.
echo Setup complete.
echo.
echo Next install Node.js LTS and Mosquitto if they are not installed.
echo.
echo Backend:
echo   .venv\Scripts\activate
echo   python -m uvicorn backend.main:app --reload
echo.
echo MQTT:
echo   mosquitto -v
echo.
echo ESP32 simulator:
echo   .venv\Scripts\activate
echo   python hardware_sim\simulated_esp32.py
echo.
echo Frontend:
echo   cd frontend
echo   npm install
echo   npm run dev
echo.
pause
