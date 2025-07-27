@echo off
echo ========================================
echo FAF5.7 Supply Chain Analysis Environment Setup
echo ========================================

echo.
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv faf5_env

echo.
echo Activating virtual environment...
call faf5_env\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Verifying installation...
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, xgboost; print('All packages installed successfully!')"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   faf5_env\Scripts\activate.bat
echo.
echo To start Jupyter Notebook, run:
echo   jupyter notebook
echo.
echo Press any key to continue...
pause 