@echo off
setlocal enabledelayedexpansion

@rem OpenCVのパス
path C:\OpenCV\ver2.4.9\build\x64\vc11\bin

@rem 評価したいサンプルの名前が記述されているテキスト
set dataset=data\ilsvrc2012_val.txt

@rem 評価したいサンプルが存在するパス
set basepath=data\val

@rem ILSVRC2012で使用する平均ファイル
set mean_path=data\ilsvrc_2012_mean.npy

@rem パラメータが保存されているフォルダ
set param_folder=params

@rem 推定結果を出力するパス
set save_folder=result

x64\Release\DLBB.exe  %dataset%  %basepath%  %mean_path%  %param_folder%  %save_folder%

pause