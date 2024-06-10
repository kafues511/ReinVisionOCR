call .venv\scripts\activate.bat

rem utf-8でマルチバイト文字列を扱えるようにします。
@chcp 65001 > nul

:MAIN

set args=
set /P args="train.py "

echo.

py -m cmds.train %args%

echo.

goto MAIN
