#pragma once
#include <algorithm>
#include <cmath>
#include "utils/tensor.hpp"


template <typename T>
class Tanh
{
public:
    typedef T InT;
    typedef T OutT;

    OutT forward(const InT & src)
    {
        std::vector<double> data = src.data;
        std::transform(
            data.begin(),
            data.end(),
            data.begin(),
            [](const double value)
            {
                return std::tanh(value);
            }
        );
        return OutT(data);
    }
};
