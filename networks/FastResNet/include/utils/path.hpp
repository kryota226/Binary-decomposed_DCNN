#pragma once
#include <string>
#include <vector>


namespace path
{

/*
dirname
--------------------------------------------------
ファイルパスからパスだけを返す。
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
ファイルパスからファイル名を返す
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
ファイルパスをパスとファイル名に分割する
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
ファイルパスから拡張子で分割して前半だけ返す。
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
パスの連結。Python の os.path と同じように使える。
ただし連結できるパスは4個まで。
*/
std::string join(
    const std::string & path1,
    const std::string & path2,
    const std::string & path3="",
    const std::string & path4="",
    const std::string & delim="/"
);

}
