#pragma once
#include <cmath>
#include "utils/math.hpp"
#include "utils/tensor.hpp"


template <typename T>
class Normalize
{
public:
    typedef T InT;
    typedef T OutT;

    OutT forward(const InT & src) const
    {
//        const OutT diff = src - src.mean();
//        const InT squared = diff.square();
//        const double stddev = std::sqrt(squared.mean());
        const InT diff = src - math::mean(src);
        const InT squared = math::square(diff);
        const double stddev = std::sqrt(math::mean(squared));
        return diff / stddev;
    }
};