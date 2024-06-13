call .venv\scripts\activate.bat

rem utf-8でマルチバイト文字列を扱えるようにします。
@chcp 65001 > nul

:MAIN

set args=
set /P args="fast_remove.py "

echo.

py -m cmds.fast_remove %args%

echo.

goto MAIN
