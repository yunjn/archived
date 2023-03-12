@echo off
title = GitBat
chcp 65001
cls
set /p name=仓库名：
cd ..
if  exist %name% goto YES
if not exist %name% goto NO

:YES
echo 仓库已存在~
pause
call Run.bat
exit

:NO
cd core
echo %name%>>r.log
cd ..
md %name%
cd %name%
cls
set /p RURL=仓库克隆地址(若无填no)：
echo %RURL%>%name%.dat
echo %name%.dat>.gitignore
echo .gitignore>>.gitignore
git init
cd ..
call Run.bat
exit