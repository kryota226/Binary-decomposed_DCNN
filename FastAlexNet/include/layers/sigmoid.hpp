#pragma once
#include <algorithm>
#include <cmath>
#include "utils/tensor.hpp"


template <typename T>
class Sigmoid
{
public:
    typedef T InT;
    typedef T OutT;

    OutT forward(const InT & src) const
    {
        std::vector<double> & data = src.data;
        std::transform(
            data.begin(), data.end(), data.begin(),
            [](const double value)
            {
                return 1.0 / (1.0 + std::exp(-value));
            }
        );
        return OutT(data);
    }
};