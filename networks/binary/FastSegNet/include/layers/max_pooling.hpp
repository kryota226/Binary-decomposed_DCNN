#pragma once
#include <algorithm>
#include "utils/tensor.hpp"


template <
    typename InT,
    int N_POOL_MAPS=1,
    int N_ROWS_KERNEL=2,
    int N_COLS_KERNEL=2,
    int N_ROWS_STRIDE=2,
    int N_COLS_STRIDE=2
>
class MaxPooling
{
public:
    typedef Tensor<
        double,
        InT::DIM_1 / N_POOL_MAPS,
        InT::DIM_2 / N_ROWS_STRIDE,
        InT::DIM_3 / N_COLS_STRIDE
    > OutT;

    OutT forward(const InT & src) const
    {
        OutT result;
        for(int d1 = 0; d1 < OutT::DIM_1; ++d1) {
            const int s1_begin = d1 * N_POOL_MAPS;
            for(int d2 = 0; d2 < OutT::DIM_2; ++d2) {
                const int s2_begin = d2 * N_ROWS_STRIDE;
                for(int d3 = 0; d3 < OutT::DIM_3; ++d3) {
                    double & dst = result.at(d1, d2, d3);
                    const int s3_begin = d3 * N_COLS_STRIDE;
                    dst = src.at(s1_begin, s2_begin, s3_begin);
                    for(int s1 = s1_begin; s1 < s1_begin + N_POOL_MAPS; ++s1) {
                        for(int s2 = s2_begin; s2 < s2_begin + N_ROWS_KERNEL; ++s2) {
                            for(int s3 = s3_begin; s3 < s3_begin + N_COLS_KERNEL; ++s3) {
                                dst = std::max(dst, src.at(s1, s2, s3));
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
};