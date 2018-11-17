#pragma once
#include "utils/tensor.hpp"


template <typename InT, typename OutT>
class Reshape
{
public:
    OutT forward(const InT & src) const
    {
        return OutT(src.data);
    }
};