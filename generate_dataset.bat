call .venv\scripts\activate.bat

rem utf-8でマルチバイト文字列を扱えるようにします。
@chcp 65001 > nul

:MAIN

set args=
set /P args="generate_dataset.py "

echo.

py -m cmds.generate_dataset %args%

echo.

goto MAIN
