#pragma once
#include "utils/tensor.hpp"


template <
    typename InT,
    typename Indices,
    int SCALE,
    int OUT_H=-1,
    int OUT_W=-1
>
class Unpooling
{
public:
    typedef Tensor<
        double,
        InT::DIM_1,
        (0 < OUT_H) ? OUT_H : 2 * InT::DIM_2,
        (0 < OUT_W) ? OUT_W : 2 * InT::DIM_3
    > OutT;

    OutT forward(const InT & src, const Indices& indices)
    {
        OutT result;
        for(int s1 = 0, d1 = 0; s1 < InT::DIM_1; ++s1, ++d1) {
            for (int s2 = 0; s2 < InT::DIM_2; ++s2) {
                for (int s3 = 0; s3 < InT::DIM_3; ++s3) {
                    const int d2 = indices.at(s1, s2, s3, 0);
                    const int d3 = indices.at(s1, s2, s3, 1);
                    result.at(d1, d2, d3) = src.at(s1, s2, s3);
                }
            }
        }
        return result;
    }
};