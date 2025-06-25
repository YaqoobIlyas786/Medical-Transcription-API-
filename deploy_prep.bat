@echo off
echo ========================================
echo Medical Transcription API - Deployment
echo ========================================
echo.
echo This script will help you prepare for deployment to Render.
echo.

echo 1. Checking Python version...
python --version

echo.
echo 2. Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo 3. Testing the application locally...
echo Starting server for 10 seconds...
timeout /t 5 /nobreak >nul
start /min python main.py
timeout /t 10 /nobreak >nul

echo.
echo 4. Project files ready for deployment:
dir /b

echo.
echo ========================================
echo Ready for Render Deployment!
echo ========================================
echo.
echo Next steps:
echo 1. Push your code to GitHub
echo 2. Connect GitHub repo to Render
echo 3. Set environment variables in Render
echo 4. Deploy!
echo.
pause
