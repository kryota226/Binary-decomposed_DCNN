#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "utils/numpy.hpp"


namespace io
{


template <typename T>
std::vector<T> load(const std::string & filename)
{
    std::cout << "loading param: " << filename << std::endl;
    try {
        std::vector<T> data;
        aoba::LoadArrayFromNumpy<T>(filename, data);
        return data;
    }
    catch(...) {
        std::cerr
            << "  ==> failed to open a file.\n"
            << std::endl;
        std::exit(-1);
    }
}

template <typename T>
void load(const std::string & filename, std::vector<T> & data)
{
    std::cout << "loading param: " << filename << std::endl;
    try {
        aoba::LoadArrayFromNumpy<T>(filename, data);
    }
    catch(...) {
        std::cerr
            << "  ==> failed to open a file.\n"
            << std::endl;
        std::exit(-1);
    }
}


void save(
    const std::string & filename,
    const std::vector<int> & evaluate,
    const int order=1,
    const bool app_flag=false
);

void save(
    const std::string & filename,
    const std::vector<double> & evaluate,
    const int order=1,
    const bool app_flag=false
);

void save(
    const std::string & filename,
    const double run_time,
    const bool app_flag=false
);

}
