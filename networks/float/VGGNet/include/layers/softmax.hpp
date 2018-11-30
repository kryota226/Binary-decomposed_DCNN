#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include "utils/tensor.hpp"


template <typename T>
class SoftMax
{
public:
    typedef T InT;
    typedef T OutT;

    OutT forward(const InT & src)
    {
        std::vector<double> exp_src = src.data;
        std::transform(
            exp_src.begin(),
            exp_src.end(),
            exp_src.begin(),
            [](const double value)
            {
                return std::exp(value);
            }
        );
        const double exp_sum
            = std::accumulate(exp_src.begin(), exp_src.end(), 0.0);
        return OutT(exp_src) / exp_sum;
    }
};