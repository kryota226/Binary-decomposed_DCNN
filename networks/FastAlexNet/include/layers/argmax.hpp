#pragma once
#include <algorithm>
#include "utils/tensor.hpp"


template <typename InT>
class ArgMax
{
public:
    typedef Tensor<int, 1> OutT;

    OutT forward(const InT & src) const
    {
        const std::vector<double> & probs = src.data;
        const std::vector<int> predict_class(
            1, std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
        );
        return OutT(predict_class);
    }
};