#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include "utils/math.hpp"
#include "utils/tensor.hpp"


template <typename T>
class SoftMax
{
public:
    typedef T InT;
    typedef T OutT;

    OutT forward(const InT & src) const
    {
        const InT exp_src = math::exp(src);
        return exp_src / math::sum(exp_src);
    }
};