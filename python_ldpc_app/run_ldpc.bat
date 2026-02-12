@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
REM LDPC Application - Batch файл для запуска
REM Использование: run_ldpc.bat [параметры]

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

REM Путь к файлу матрицы по умолчанию (относительно корня проекта)
set DEFAULT_MATRIX=..\Channel_Codes_Database\BCH_7_4_1_strip.alist.txt

REM Если передан первый аргумент как путь к матрице, используем его
if "%~1"=="" (
    set MATRIX_FILE=%DEFAULT_MATRIX%
) else (
    set MATRIX_FILE=%~1
)

REM Метод кодирования (по умолчанию: standard)
REM Доступные значения: standard, richardson-urbanke
if "%~2"=="" (
    set ENCODING_METHOD=standard
) else (
    set ENCODING_METHOD=%~2
)

REM Количество потоков (по умолчанию: 1)
REM Для многопоточной обработки укажите число потоков (например, 4)
if "%~3"=="" (
    set THREADS=1
) else (
    set THREADS=%~3
)

REM Gap для метода Ричардсона-Урбанке (по умолчанию: не задан, автоматический поиск)
REM Укажите число для задания конкретного gap (например, 0, 2, 5)
if "%~4"=="" (
    set RU_GAP=
) else (
    set RU_GAP=%~4
)

REM Проверяем существование файла матрицы
if not exist "%MATRIX_FILE%" (
    echo Ошибка: Файл матрицы не найден: %MATRIX_FILE%
    echo.
    echo Использование:
    echo   run_ldpc.bat [путь_к_матрице] [метод_кодирования] [количество_потоков] [ru_gap]
    echo.
    echo Примеры:
    echo   run_ldpc.bat
    echo   run_ldpc.bat ..\Channel_Codes_Database\BCH_7_4_1_strip.alist.txt
    echo   run_ldpc.bat ..\Channel_Codes_Database\BCH_7_4_1_strip.alist.txt standard
    echo   run_ldpc.bat ..\Channel_Codes_Database\BCH_7_4_1_strip.alist.txt richardson-urbanke
    echo   run_ldpc.bat ..\Channel_Codes_Database\BCH_7_4_1_strip.alist.txt standard 4
    echo   run_ldpc.bat ..\Channel_Codes_Database\BCH_7_4_1_strip.alist.txt richardson-urbanke 4 2
    echo.
    echo Методы кодирования:
    echo   standard - стандартный метод (по умолчанию)
    echo   richardson-urbanke - метод Ричардсона-Урбанке
    echo.
    echo Количество потоков:
    echo   1 - однопоточный режим (по умолчанию)
    echo   2, 4, 8 и т.д. - многопоточный режим для ускорения обработки
    echo.
    echo Gap для метода Ричардсона-Урбанке:
    echo   не указан - автоматический поиск минимального gap
    echo   0 - стандартная форма [A | I_m]
    echo   2, 5, 10 и т.д. - заданный gap
    pause
    exit /b 1
)

echo ========================================
echo LDPC Application - Запуск приложения
echo ========================================
echo Матрица: %MATRIX_FILE%
echo Метод кодирования: %ENCODING_METHOD%
echo Количество потоков: %THREADS%
if defined RU_GAP (
    if not "!RU_GAP!"=="" (
        echo Gap (РУ): !RU_GAP!
    )
)
echo.

REM Запускаем приложение с параметрами
set "PYTHON_CMD=python main.py --matrix "%MATRIX_FILE%" --encoding-method %ENCODING_METHOD% --threads %THREADS%"
if defined RU_GAP (
    if not "!RU_GAP!"=="" (
        set "PYTHON_CMD=!PYTHON_CMD! --ru-gap !RU_GAP!"
    )
)
set "PYTHON_CMD=!PYTHON_CMD! --ber --fer"
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
