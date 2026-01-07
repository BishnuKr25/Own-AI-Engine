@echo off
echo ========================================
echo Sovereign AI Suite - Windows Installation
echo ========================================

:: Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed! Please install Python 3.10 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Create project directory
echo Creating project directory...
if not exist "C:\sovereign-ai-suite" mkdir "C:\sovereign-ai-suite"
cd /d "C:\sovereign-ai-suite"

:: Create subdirectories
mkdir backend backend\core backend\services backend\utils
mkdir admin admin\templates
mkdir frontend
mkdir models
mkdir data
mkdir logs
mkdir config

:: Create virtual environment
echo Creating Python virtual environment...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r setup\requirements.txt

:: Download models (this will take time)
echo Downloading AI models...
python setup\download_models.py

echo Installation complete!
echo To start the application, run: start_services.bat
pause