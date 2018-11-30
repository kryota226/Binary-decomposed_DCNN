#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>
#include "utils/io.hpp"
#include "utils/numpy.hpp"
#include "utils/path.hpp"


namespace io
{

void save(
    const std::string & filename,
    const std::vector<int> & evaluate,
    const int order,
    const bool output_status
) {
    std::tr2::sys::create_directories(std::tr2::sys::path(path::dirname(filename))); // mkdir
    std::string predict = std::to_string(evaluate[0]);
    for(int i = 1; i < order; ++i) {
        predict.append(", " + std::to_string(evaluate[i]));
    }
    const auto status = (true == output_status) ? std::ios::app : std::ios::out;
    std::ofstream file(filename, status);
    if(!file.is_open()) {
        std::cerr << "file open to a failed." << std::endl;
        std::exit(-1);
    }
    file << predict << std::endl;
}

void save(
    const std::string & filename,
    const std::vector<double> & evaluate,
    const int order,
    const bool output_status
) {
    std::tr2::sys::create_directories(std::tr2::sys::path(path::dirname(filename))); // mkdir
    std::string scores = std::to_string(evaluate[0]);
    for(int i = 1; i < order; ++i) {
        scores.append(", " + std::to_string(evaluate[i]));
    }
    const auto status = (true == output_status) ? std::ios::app : std::ios::out;
    std::ofstream file(filename, status);
    if(!file.is_open()) {
        std::cerr << "file open to a failed." << std::endl;
        std::exit(-1);
    }
    file << scores << std::endl;
}

void save(
    const std::string & filename,
    const double run_time,
    const bool app_flag
) {
    std::tr2::sys::create_directories(std::tr2::sys::path(path::dirname(filename))); // mkdir
    const auto app = (true == app_flag) ? std::ios::app : std::ios::out;
    std::ofstream file(filename, app);
    if(!file.is_open()) {
        std::cerr << "file open to a failed." << std::endl;
        std::exit(-1);
    }
    file << run_time << std::endl;
}

}
