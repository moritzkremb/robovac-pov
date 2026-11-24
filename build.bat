@echo off
rem build.bat - Windows/CUDA surumunu derler (robovac.exe). Mac surumu icin build.sh.
setlocal
cd /d "%~dp0"
where cl >nul 2>nul
if not errorlevel 1 goto havecl
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`call "%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
:havecl
set "NVCC=nvcc"
if defined CUDA_PATH set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
"%NVCC%" -O3 -std=c++17 -arch=sm_86 --use_fast_math -Xcompiler "/O2,/openmp,/EHsc" -o robovac.exe src\main_cuda.cpp src\render.cu
if errorlevel 1 exit /b 1
echo built robovac.exe
