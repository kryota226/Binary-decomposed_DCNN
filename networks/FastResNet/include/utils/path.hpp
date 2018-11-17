#pragma once
#include <string>
#include <vector>


namespace path
{

/*
dirname
--------------------------------------------------
�t�@�C���p�X����p�X������Ԃ��B
*/
std::string dirname(
    const std::string & src_path
);

void dirname(
    const std::string & src_path,
    std::string & dst
);


/*
basename
--------------------------------------------------
�t�@�C���p�X����t�@�C������Ԃ�
*/
std::string basename(
    const std::string & src_path
);

void basename(
    const std::string & src_path,
    std::string & dst
);


/*
split
--------------------------------------------------
�t�@�C���p�X���p�X�ƃt�@�C�����ɕ�������
*/
std::vector<std::string> split(
    const std::string & src_path
);

void split(
    const std::string & src_path,
    std::vector<std::string> & dst_path
);

void split(
    const std::string & src_path,
    std::string & dirname,
    std::string & basename
);


/*
splitext
--------------------------------------------------
�t�@�C���p�X����g���q�ŕ������đO�������Ԃ��B
*/
std::string splitext(
    const std::string & src_path
);

void splitext(
    const std::string & src_path,
    std::string & dst_path
);


/*
join
--------------------------------------------------
�p�X�̘A���BPython �� os.path �Ɠ����悤�Ɏg����B
�������A���ł���p�X��4�܂ŁB
*/
std::string join(
    const std::string & path1,
    const std::string & path2,
    const std::string & path3="",
    const std::string & path4="",
    const std::string & delim="/"
);

}
