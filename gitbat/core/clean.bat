@echo off
title = GitBat
if exist trash rd /s/q trash
echo.|md trash
call Run.bat
exit