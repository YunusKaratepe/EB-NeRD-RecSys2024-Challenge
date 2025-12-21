@echo off
REM Helper script to run the winner's solution without Docker
REM Usage: run.bat <command> [--debug]
REM Example: run.bat create-candidates --debug
REM Example: run.bat create-features --debug
REM Example: run.bat create-datasets --debug

cd /d "%~dp0"
set PYTHONPATH=%~dp0
call conda activate "../../.conda"
".conda/python.exe" -m invoke %*
