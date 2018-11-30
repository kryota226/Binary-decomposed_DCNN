#pragma once
#include "utils/tensor.hpp"


template <typename T>
class Linear
{
public:
    typedef T InT;
    typedef T OutT;

    Linear(const double multiplier = 1.0, const double accumulator = 0.0)
        : multiplier_(multiplier), accumulator_(accumulator)
    {}

    OutT forward(const InT & src)
    {
        return src * multiplier_ + accumulator_;
    }

private:
    double multiplier_;
    double accumulator_;
};