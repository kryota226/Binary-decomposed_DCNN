#pragma once
#include "utils/tensor.hpp"


template <typename InT>
class Flatten
{
public:
    typedef Tensor<double, InT::SIZE> OutT;

    OutT forward(const InT & src) const
    {
        return OutT(src.data);
    }
};