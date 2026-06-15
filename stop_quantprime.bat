@echo off
setlocal

echo ============================================================
echo  QuantPrime Stop Helper
echo ============================================================
echo.
echo This will try to stop common QuantPrime local processes:
echo - node.exe
echo - python.exe
echo.
echo WARNING: This may also stop other Node/Python processes.
echo Close this window if you do not want that.
echo.
pause

taskkill /F /IM node.exe
taskkill /F /IM python.exe

echo.
echo Done.
pause
endlocal
