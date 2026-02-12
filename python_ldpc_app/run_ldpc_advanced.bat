@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
REM LDPC Application - Расширенный batch файл для запуска с настройками
REM Использование: run_ldpc_advanced.bat

REM Получаем путь к директории, где находится bat файл
cd /d "%~dp0"

REM Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Ошибка: Python не найден в PATH
    echo Убедитесь, что Python установлен и добавлен в PATH
    pause
    exit /b 1
)
wimax_576_0.83.alist
REM Настройки по умолчанию
set MATRIX_FILE=..\Channel_Codes_Database\Custom LDPC Codes\CCSDS_ldpc_n32_k16.alist.txt
set BLOCKS=1000
set ITERATIONS=5
set INITIAL_SNR=0.0
set END_SNR=5.0
set STEP_SNR=1.0
set INTERLEAVER=none
set ENCODING_METHOD=standard
set RU_GAP=
set THREADS=8
set MODE=1
set INTERFERENCE_SNR=1.0
set P=0.1
set MODULATION=1

REM Можно изменить параметры здесь:
REM set BLOCKS=50
REM set INITIAL_SNR=1.0
REM set END_SNR=4.0
REM set STEP_SNR=0.5
REM set INTERLEAVER=random
REM set ENCODING_METHOD=richardson-urbanke
REM set RU_GAP=2
REM set THREADS=4
REM set MODE=2

echo ========================================
echo LDPC Application - Расширенный запуск
echo ========================================
echo Параметры:
echo   Матрица: %MATRIX_FILE%
echo   Блоков: %BLOCKS%
echo   Итераций: %ITERATIONS%
echo   SNR: %INITIAL_SNR% - %END_SNR% дБ (шаг: %STEP_SNR%)
echo   Интерливер: %INTERLEAVER%
echo   Метод кодирования: %ENCODING_METHOD%
set "GAP_DISPLAY=автоматический поиск"
if defined RU_GAP (
    if not "!RU_GAP!"=="" (
        set "GAP_DISPLAY=!RU_GAP!"
    )
)
echo   Gap (РУ): !GAP_DISPLAY!
echo   Количество потоков: %THREADS%
echo   Режим канала: %MODE%
echo ========================================
echo.

REM Проверяем существование файла матрицы
if not exist "%MATRIX_FILE%" (
    echo Ошибка: Файл матрицы не найден: %MATRIX_FILE%
    pause
    exit /b 1
)

REM Запускаем приложение с параметрами
set "PYTHON_CMD=python main.py --matrix "%MATRIX_FILE%" --blocks %BLOCKS% --iterations %ITERATIONS% --initial-snr %INITIAL_SNR% --end-snr %END_SNR% --step-snr %STEP_SNR% --interleaver %INTERLEAVER% --encoding-method %ENCODING_METHOD% --threads %THREADS% --mode %MODE% --interference-snr %INTERFERENCE_SNR% --p %P% --modulation %MODULATION%"
if defined RU_GAP (
    if not "!RU_GAP!"=="" (
        set "PYTHON_CMD=!PYTHON_CMD! --ru-gap !RU_GAP!"
    )
)
set "PYTHON_CMD=!PYTHON_CMD! --ber --fer --normalized-llr"
!PYTHON_CMD!

REM Если произошла ошибка, показываем сообщение
if errorlevel 1 (
    echo.
    echo ========================================
    echo Произошла ошибка при выполнении
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Выполнение завершено успешно
echo ========================================
pause
