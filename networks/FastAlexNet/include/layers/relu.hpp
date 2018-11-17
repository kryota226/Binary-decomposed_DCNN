#pragma once
#include <algorithm>
#include "utils/tensor.hpp"


template <typename T>
class ReLU
{
public:
    typedef T InT;
    typedef T OutT;

    OutT forward(const InT & src) const
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
    }
};