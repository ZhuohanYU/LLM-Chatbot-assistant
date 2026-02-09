@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Starting Flask application...
echo Note: First run will download the GPT-Neo-1.3B model (~5GB) which may take several minutes.
echo.
python app.py
pause


