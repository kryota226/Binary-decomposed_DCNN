@echo off

@rem OpenCV�̃p�X
path C:\OpenCV\ver2.4.9\build\x64\vc11\bin

@rem �]���������T���v���̖��O���L�q����Ă���e�L�X�g
set dataset=data\ilsvrc2012_val.txt

@rem �]���������T���v�������݂���p�X
set basepath=data\val

@rem �p�����[�^���ۑ�����Ă���t�H���_
set param_folder=params

@rem ���茋�ʂ��o�͂���p�X
set save_folder=result

x64\Release\DLBB.exe  %dataset%  %basepath%  %param_folder%  %save_folder%

pause