@echo off

@rem OpenCVのパス
path C:\OpenCV\ver2.4.9\build\x64\vc11\bin

@rem 評価したいサンプルの名前が記述されているテキスト
set dataset=E:\data\imagenet\ilsvrc2012_valid.txt

@rem 評価したいサンプルが存在するパス
set basepath=E:\data\imagenet\val

@rem パラメータが保存されているフォルダ
set param_folder=params

@rem 推定結果を出力するパス
set save_folder=result_50000

x64\Release\DLBB.exe  %dataset%  %basepath%  %param_folder%  %save_folder%

pause