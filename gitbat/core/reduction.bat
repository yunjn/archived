@echo off
title = GitBat
chcp 65001
cls
if not exist trash md trash
echo.|cd trash
mode con cols=56 lines=30
echo.
echo *************************回收站*************************
dir /d/l trash
echo ********************************************************
set /p name=仓库名：
::echo.|cd ..
echo.|cd ..
move trash/%name% ..\
cd ..
call Run.bat
exit