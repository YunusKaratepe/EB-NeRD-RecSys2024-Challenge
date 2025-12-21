@echo off
REM Run experiment with multiple seeds for significance analysis
REM Usage: run_multiple_seeds.bat <exp_name> <seed1> <seed2> <seed3>
REM Example: run_multiple_seeds.bat medium067_001 7 42 123

if "%1"=="" (
    echo Usage: run_multiple_seeds.bat ^<exp_name^> ^<seed1^> ^<seed2^> ^<seed3^>
    echo Example: run_multiple_seeds.bat medium067_001 7 42 123
    exit /b 1
)

set EXP_NAME=%1
set SEED1=%2
set SEED2=%3
set SEED3=%4

if "%SEED1%"=="" set SEED1=7
if "%SEED2%"=="" set SEED2=42
if "%SEED3%"=="" set SEED3=123

echo ========================================
echo Running %EXP_NAME% with seeds: %SEED1%, %SEED2%, %SEED3%
echo ========================================

echo.
echo [1/3] Running with seed=%SEED1%
call run.bat train --exp=%EXP_NAME% --seed=%SEED1%
if errorlevel 1 (
    echo ERROR: Failed with seed=%SEED1%
    exit /b 1
)

echo.
echo [2/3] Running with seed=%SEED2%
call run.bat train --exp=%EXP_NAME% --seed=%SEED2%
if errorlevel 1 (
    echo ERROR: Failed with seed=%SEED2%
    exit /b 1
)

echo.
echo [3/3] Running with seed=%SEED3%
call run.bat train --exp=%EXP_NAME% --seed=%SEED3%
if errorlevel 1 (
    echo ERROR: Failed with seed=%SEED3%
    exit /b 1
)

echo.
echo ========================================
echo All seeds completed successfully!
echo Check output directories for results with different seeds
echo ========================================
