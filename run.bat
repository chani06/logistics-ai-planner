@echo off
chcp 65001 >nul
echo.
echo =========================================
echo   Trip Planner - Network Server
echo =========================================
echo.

REM หา IP จริงในเครือข่าย (กรองเฉพาะ 10.x หรือ 192.168.x หรือ 172.x)
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "172.21"') do (
    set LOCAL_IP=%%A
)
REM ตัดช่องว่างหน้า
set LOCAL_IP=%LOCAL_IP: =%

echo  เครื่องนี้  : http://localhost:8501
echo  เครือข่าย  : http://%LOCAL_IP%:8501
echo.
echo  เครื่องอื่นในวง LAN เดียวกันเปิด browser แล้วไปที่:
echo  http://%LOCAL_IP%:8501
echo.
echo  กด Ctrl+C เพื่อหยุด server
echo =========================================
echo.

streamlit run app.py

pause
