@echo off
setlocal enabledelayedexpansion

@rem OpenCV�̃p�X
path C:\OpenCV\ver2.4.9\build\x64\vc11\bin

@rem �]���������T���v���̖��O���L�q����Ă���e�L�X�g
set dataset=data\ilsvrc2012_val.txt

@rem �]���������T���v�������݂���p�X
set basepath=data\val

@rem ILSVRC2012�Ŏg�p���镽�σt�@�C��
set mean_file=data\ilsvrc_2012_mean.npy

@rem ���茋�ʂ��o�͂���p�X
set save_folder=result

x64\Release\DLBB.exe  %dataset%  %basepath%  %mean_file%  %save_folder%

pause