call .venv\scripts\activate.bat

rem utf-8でマルチバイト文字列を扱えるようにします。
@chcp 65001 > nul

:MAIN

set args=
set /P args="ckpt_to_pretrained.py "

echo.

py -m cmds.ckpt_to_pretrained %args%

echo.

goto MAIN
