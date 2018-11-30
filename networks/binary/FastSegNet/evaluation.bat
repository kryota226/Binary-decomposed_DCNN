@echo off
setlocal enabledelayedexpansion

@rem OpenCVのパス
path C:\OpenCV\ver2.4.9\build\x64\vc11\bin

@rem 評価したいサンプルの名前が記述されているテキスト
set dataset=data\cityscapes\valset10.txt

@rem 評価したいサンプルが存在するパス
set basepath=data\cityscapes

@rem パラメータが保存されているフォルダ
set param_path=params

@rem 推定結果を出力するパス
set save_folder=result

set file_path=bit6\basis6
md %save_folder%\!file_path!
x64\Release\DLBB.exe  %dataset%  %basepath%  %param_path%  %save_folder%\!file_path!

pause