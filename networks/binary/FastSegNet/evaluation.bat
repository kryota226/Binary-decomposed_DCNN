@echo off
setlocal enabledelayedexpansion

@rem OpenCV�̃p�X
path C:\OpenCV\ver2.4.9\build\x64\vc11\bin

@rem �]���������T���v���̖��O���L�q����Ă���e�L�X�g
set dataset=data\cityscapes\valset10.txt

@rem �]���������T���v�������݂���p�X
set basepath=data\cityscapes

@rem �p�����[�^���ۑ�����Ă���t�H���_
set param_path=params

@rem ���茋�ʂ��o�͂���p�X
set save_folder=result

set file_path=bit6\basis6
md %save_folder%\!file_path!
x64\Release\DLBB.exe  %dataset%  %basepath%  %param_path%  %save_folder%\!file_path!

pause