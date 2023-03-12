### <div align=center>Git的批处理管理</div>

* 新建  
新建Git仓库，提供些基础文件
* 提交  
提交到本地仓库
* 推送  
推送到远程仓库
* 同步  
* 打开  
打开仓库位置
* 重建  
删除所有log只保留最新的文件版本
* 移除  
仓库移动到core\trash
* 还原  
从trash还原
* 清理
清空trash
* ~~选项10升级~~

~~安装脚本~~  
~~@echo off~~  
~~chcp 65001~~  
~~cls~~  
~~mode con cols=56 lines=18~~  
~~title = GitBat-setup~~  
~~msg * 需要Git工具的支持哟~~~  
~~echo \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*安装\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*~~  
~~set /p pan=盘符：~~  
~~%pan%:~~  
~~cd \~~  
~~set /p pa=路径：~~  
~~if not exist %pa% md %pa%~~  
~~cd %pa%~~  
~~git clone https://github.com/jfian/GitBat.git~~  
~~echo 位置：%pan%:\%pa%~~  
~~set /p =安装完毕！<nul~~  
~~pause >nul~~  
~~cd GitBat~~  
~~explorer.exe .~~  

