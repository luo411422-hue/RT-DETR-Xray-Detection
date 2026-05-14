@echo off
REM ===== 启动 Streamlit 网页版检测系统 =====

REM 1. 激活虚拟环境（替换 my_env 为你自己的环境名）
call conda activate E:\my_env\fire

REM 2. 切换到项目目录
cd /d "E:\华北五省\参考材料\参考代码\pyqt-master"

REM 3. 启动 Streamlit
streamlit run web_app2.py

REM 4. 等待用户按键关闭窗口
pause


