@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ============================================================
echo EmoAtlas Windows setup: conda environment + Jupyter kernel
echo ============================================================
echo.

set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" set "CONDA_BAT=C:\ProgramData\Anaconda3\condabin\conda.bat"
if not exist "%CONDA_BAT%" set "CONDA_BAT=C:\ProgramData\Miniconda3\condabin\conda.bat"

if not exist "%CONDA_BAT%" (
    echo ERROR: Could not find conda.bat in the common Anaconda/Miniconda locations.
    echo.
    echo Try running this from the Anaconda Prompt, or edit this file and set CONDA_BAT
    echo to the full path of your conda.bat file.
    echo Common example:
    echo   C:\Users\YOUR_USER\anaconda3\condabin\conda.bat
    echo.
    pause
    exit /b 1
)

echo Found conda at:
echo   %CONDA_BAT%
echo.

echo Creating or updating conda environment: emoatlas311
call "%CONDA_BAT%" create -n emoatlas311 python=3.11 -y
if errorlevel 1 (
    echo Conda environment creation failed.
    pause
    exit /b 1
)

echo Activating environment...
call "%CONDA_BAT%" activate emoatlas311
if errorlevel 1 (
    echo Could not activate emoatlas311.
    pause
    exit /b 1
)

echo Upgrading packaging tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :piperror

echo Installing EmoAtlas from GitHub...
python -m pip install --no-cache-dir git+https://github.com/massimostel/emoatlas.git
if errorlevel 1 goto :piperror

echo Repairing transformers/tokenizers compatibility...
python -m pip install --no-cache-dir --force-reinstall "tokenizers>=0.21,<0.22"
if errorlevel 1 goto :piperror

echo Installing the lightweight English spaCy model...
python -m spacy download en_core_web_lg
if errorlevel 1 goto :piperror

echo Installing Jupyter kernel support...
python -m pip install --upgrade ipykernel jupyterlab notebook
if errorlevel 1 goto :piperror

echo Downloading NLTK WordNet resources...
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
if errorlevel 1 goto :piperror

echo Registering Jupyter kernel...
python -m ipykernel install --user --name emoatlas311 --display-name "Python (emoatlas311)"
if errorlevel 1 goto :piperror

echo Checking package consistency...
python -m pip check

echo.
echo ============================================================
echo Setup complete.
echo In Jupyter, choose kernel: Python (emoatlas311)
echo Then open/run: EmoAtlas_English_Text_Analysis_FIXED.ipynb
echo ============================================================
echo.
pause
exit /b 0

:piperror
echo.
echo ERROR: A Python/pip command failed. Scroll up to see the first error.
echo.
pause
exit /b 1
