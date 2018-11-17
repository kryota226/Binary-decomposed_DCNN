#include <algorithm>
#include "utils/path.hpp"


namespace path
{

/*
dirname
*/
std::string dirname(const std::string & src_path)
{
    const std::string::size_type pos = std::max<signed>(
        src_path.find_last_of('/'), src_path.find_last_of('\\')
    );
    return (pos == std::string::npos) 
        ? "" : src_path.substr(0, pos);
}

void dirname(const std::string & src_path, std::string & dst)
{
    const std::string::size_type pos = std::max<signed>(
        src_path.find_last_of('/'), src_path.find_last_of('\\')
    );
    dst = (pos == std::string::npos)
        ? "" : src_path.substr(0, pos);
}


/*
basename
*/
std::string basename(const std::string & src_path)
{
    return src_path.substr(std::max<signed>(
            src_path.find_last_of('/'), src_path.find_last_of('\\')
        ) + 1
    );
}

void basename(const std::string & src_path, std::string & dst)
{
    dst = src_path.substr(std::max<signed>(src_path.find_last_of('/'), src_path.find_last_of('\\')) + 1);
}


/*
split
*/
std::vector<std::string> split(const std::string & src_path)
{
    enum { HEAD, TAIL, NUM_SPLITS };
    std::vector<std::string> dst_path(NUM_SPLITS);
    const std::string::size_type pos = std::max<signed>(
        src_path.find_last_of('/'),
        src_path.find_last_of('\\')
    );
    dst_path[HEAD] = src_path.substr(0, pos);
    dst_path[TAIL] = src_path.substr(pos + 1);
    return dst_path;
}

void split(const std::string & src_path, std::vector<std::string> & dst_path)
{
    enum { HEAD, TAIL, NUM_SPLITS };
    dst_path = std::vector<std::string>(NUM_SPLITS);
    const std::string::size_type pos = std::max<signed>(
        src_path.find_last_of('/'),
        src_path.find_last_of('\\')
    );
    dst_path[HEAD] = src_path.substr(0, pos);
    dst_path[TAIL] = src_path.substr(pos + 1);
}

void split(const std::string & src_path, std::string & dirname, std::string & basename)
{
    const std::string::size_type pos = std::max<signed>(
        src_path.find_last_of('/'), src_path.find_last_of('\\')
    );
    dirname = src_path.substr(0, pos);
    basename = src_path.substr(pos + 1);
}


/*
splitext
*/
std::string splitext(const std::string & src_path)
{
    return src_path.substr(0, src_path.find_last_of('.'));
}

void splitext(const std::string & src_path, std::string & dst_path)
{
    dst_path = src_path.substr(0, src_path.find_last_of('.'));
}


/*
join
*/
std::string join(
    const std::string & path1,
    const std::string & path2,
    const std::string & path3,
    const std::string & path4,
    const std::string & delim
) {
    std::string path;
    path.append(path1.empty() ? "" : path1);
    path.append(path2.empty() ? "" : delim + path2);
    path.append(path3.empty() ? "" : delim + path3);
    path.append(path4.empty() ? "" : delim + path4);
    return path;
}

}
