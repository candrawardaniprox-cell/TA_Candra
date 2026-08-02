@echo off
echo ============================================================
echo   SETUP VENV_BARU - Python 3.13
echo ============================================================
echo.

set PYTHON313=C:\Users\LuDoYa\AppData\Local\Programs\Python\Python313\python.exe
set VENV_DIR=f:\TA_Candra_main_baru\venv_baru
set REQ_FILE=f:\TA_Candra_main_baru\requirements.txt

REM Cek apakah Python 3.13 tersedia
echo [1/4] Mengecek Python 3.13...
if not exist "%PYTHON313%" (
    echo ERROR: Python 3.13 tidak ditemukan di:
    echo   %PYTHON313%
    pause
    exit /b 1
)
"%PYTHON313%" --version
echo Python 3.13 ditemukan!
echo.

REM Hapus venv_baru lama jika ada
echo [2/4] Menghapus venv_baru lama...
if exist "%VENV_DIR%" (
    rmdir /s /q "%VENV_DIR%"
    if exist "%VENV_DIR%" (
        echo WARNING: Gagal hapus venv_baru - mungkin ada proses yang menggunakan folder ini.
        echo Coba tutup VS Code / Jupyter / terminal lain yang menggunakan venv_baru, lalu jalankan lagi.
        pause
        exit /b 1
    )
    echo venv_baru lama berhasil dihapus.
) else (
    echo venv_baru tidak ditemukan, langsung buat baru.
)
echo.

REM Buat venv baru
echo [3/4] Membuat venv_baru baru dengan Python 3.13...
"%PYTHON313%" -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Gagal membuat virtual environment!
    pause
    exit /b 1
)
echo venv_baru berhasil dibuat!
echo.

REM Install requirements
echo [4/4] Menginstall dependencies dari requirements.txt...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip
if exist "%REQ_FILE%" (
    pip install -r "%REQ_FILE%"
) else (
    echo WARNING: requirements.txt tidak ditemukan, install ultralytics saja...
    pip install ultralytics torch torchvision
)
echo.

echo ============================================================
echo   SELESAI! venv_baru siap digunakan.
echo ============================================================
echo.
echo Untuk mengaktifkan venv_baru, jalankan:
echo   f:\TA_Candra_main_baru\venv_baru\Scripts\activate.bat
echo.
pause
