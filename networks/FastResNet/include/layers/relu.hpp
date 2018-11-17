#pragma once
#include <algorithm>
#include "utils/tensor.hpp"


template <typename T>
class ReLU
{
public:
    typedef T InT;
    typedef T OutT;

    ReLU(void) {}

    ReLU(const std::string & map_file)
        : chainer_map(io::load<double>(map_file))
    {}

    OutT forward(const InT & src)
    {
        std::vector<double> data = src.data;
        std::transform(
            data.begin(), data.end(), data.begin(),
            [](const double value)
            {
                return std::max(value, 0.0);
            }
        );
        return OutT(data);

        /*
        if(!chainer_map.empty()) {
#include "utils/io.hpp"
            if(data.size() == chainer_map.size()) {
                double abs_error = 0.0;
                for(int i = 0; i < data.size(); ++i) {
                    std::cout << data[i] << std::endl;
                    std::cout << chainer_map[i] << std::endl;
                    getchar();
                    abs_error += std::fabs(chainer_map[i] - data[i]);
                }
                std::cout << "|cpp - chainer|.sum() = " << abs_error << std::endl;
                std::cout << "|cpp - chainer|.mean() = " << abs_error / data.size() << std::endl;
                std::cout << "stop." << std::endl;
                getchar();
            }
            else {
                std::cerr << "Not matched cpp.size() and chainer_map.size()." << std::endl;
                getchar();
                std::exit(-1);
            }
        }
        return OutT(data);
        */
    }

private:
    std::vector<double> chainer_map;
};