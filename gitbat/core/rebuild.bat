@echo off
title = GitBat
chcp 65001
cls
set /p name=仓库名：

cd ..
if  exist %name% goto YES
if not exist %name% goto NO

:YES
cd %name%
echo.|rd /s/q .git 
echo.|git init
echo.|git add .
echo.|git commit -m"clean notes"
set /p RURL=<%name%.dat
if %RURL%==no goto EX
echo.|git remote add origin %RURL%
echo.|git push origin master --force
git push --set-upstream origin master
:EX
cd ..
call Run.bat
exit

:NO
echo 仓库不存在
pause
call Run.bat
exit