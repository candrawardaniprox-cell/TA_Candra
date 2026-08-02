@echo off
echo ============================================================
echo   AKTIFKAN VENV_BARU DAN JALANKAN TRAINING
echo ============================================================
echo.

set VENV_DIR=f:\TA_Candra_main_baru\venv_baru
set TRAIN_SCRIPT=f:\TA_Candra_main_baru\train.py

REM Cek venv_baru
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: venv_baru tidak ditemukan atau belum dibuat!
    echo Jalankan dulu: setup_venv_baru.bat
    pause
    exit /b 1
)

echo Mengaktifkan venv_baru...
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo Python aktif:
python --version
echo Path:
where python

echo.
echo ============================================================
echo   Memulai Training...
echo ============================================================
echo.

if exist "%TRAIN_SCRIPT%" (
    python "%TRAIN_SCRIPT%"
) else (
    echo ERROR: train.py tidak ditemukan di:
    echo   %TRAIN_SCRIPT%
    echo.
    echo Masuk ke direktori project:
    cd /d f:\TA_Candra_main_baru
    echo.
    echo Ketik perintah training Anda secara manual.
    cmd /k
)

echo.
pause
