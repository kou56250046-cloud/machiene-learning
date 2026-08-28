@echo off
setlocal
title ML Lab

rem ====================================================================
rem  ML Lab 起動スクリプト
rem
rem  このファイルをダブルクリックすると、アプリが起動してブラウザが開きます。
rem  終了するときは、この黒い画面を閉じてください。
rem
rem  プロジェクトを別の場所へ移した場合は、下の PROJECT= を書き換えてください。
rem ====================================================================

set "PROJECT=C:\Users\kou56\projects\machiene-learning"
set "PORT=8501"
set "URL=http://localhost:%PORT%"
set "SYS=%SystemRoot%\System32"

echo.
echo   ML Lab
echo   ==================================================
echo.

rem --- プロジェクトの場所を確認 ---------------------------------------
if not exist "%PROJECT%\app\Home.py" (
    echo   [エラー] プロジェクトが見つかりません。
    echo.
    echo     探した場所: %PROJECT%
    echo.
    echo   このファイルを右クリック ^> 編集 で開き、
    echo   PROJECT= の行を正しいフォルダに書き換えてください。
    echo.
    pause
    exit /b 1
)

cd /d "%PROJECT%"

rem --- uv があるか確認 -------------------------------------------------
where uv >nul 2>&1
if errorlevel 1 (
    echo   [エラー] uv が見つかりません。
    echo.
    echo   PowerShell を開いて次を実行し、PC を再起動してからお試しください:
    echo     irm https://astral.sh/uv/install.ps1 ^| iex
    echo.
    pause
    exit /b 1
)

rem --- すでに起動していないか確認 --------------------------------------
"%SYS%\netstat.exe" -ano | "%SYS%\findstr.exe" /r /c:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo   すでに起動しています。ブラウザを開きます。
    echo.
    start "" "%URL%"
    "%SYS%\ping.exe" -n 4 127.0.0.1 >nul
    exit /b 0
)

rem --- 起動 ------------------------------------------------------------
echo   起動しています... 初回は少し時間がかかります。
echo.
echo   ブラウザが自動で開きます。開かない場合はこちらへ:
echo     %URL%
echo.
echo   終了するときは、この画面を閉じてください。
echo   ==================================================
echo.

rem サーバーが立ち上がるのを待ってからブラウザを開く
start "" /b cmd /c ""%SYS%\ping.exe" -n 9 127.0.0.1 >nul ^& start "" "%URL%""

uv run streamlit run app/Home.py --server.port %PORT% --server.headless true

rem --- サーバーが止まったらここに来る ---------------------------------
echo.
echo   ==================================================
echo   ML Lab を終了しました。
echo.
pause
