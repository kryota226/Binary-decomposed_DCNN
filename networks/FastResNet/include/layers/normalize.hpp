#pragma once
#include <cmath>
#include "utils/tensor.hpp"


template <typename T>
class Normalize
{
public:
    typedef T InT;
    typedef T OutT;

    OutT forward(const InT & src)
    {
        const OutT diff(src - src.mean());
        const InT squared = diff.square();
        const double stddev = std::sqrt(squared.mean());
        return diff / stddev;
    }
};