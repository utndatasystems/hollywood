@echo off
setlocal

if "%PYTHON_EXE%"=="" set PYTHON_EXE=python
set SCRIPT_DIR=%~dp0
set PYTHONUNBUFFERED=1

"%PYTHON_EXE%" -u "%SCRIPT_DIR%run_full_api_pipeline.py" %*
exit /b %ERRORLEVEL%
