@echo off
title = GitBat
mode con cols=56 lines=18
%~d0
cd %~dp0/core
if not exist r.log call sread.bat
call:wel
set /p sel=
if %sel%==1 call init.bat 
if %sel%==2 call add.bat
if %sel%==3 call push.bat
if %sel%==4 call sync.bat
if %sel%==5 call open.bat
if %sel%==6 call rebuild.bat
if %sel%==7 call move.bat
if %sel%==8 call reduction.bat
if %sel%==9 call clean.bat
if %sel%==10 call update.bat
:wel
chcp 65001
cls
echo.
echo.                     
echo                     +              +           
echo         ==========/===\==========/===\==========
echo         *                                      *
echo         *                GitNote               *
echo         *                                      *
echo         *      1.新建    2.提交    3.推送      *
echo         *                                      *
echo         *      4.同步    5.打开    6.重建      *
echo         *                                      *
echo         *      7.移除    8.还原    9.清理      *
echo         *                                      *
echo         * V1.0.3            ‿           By IAN *
echo         +--------------------------------------+
goto:eof


