#pragma once
#include "utils/tensor.hpp"


template <
    typename InT,
    int N_ROWS_KERNEL = 2,
    int N_COLS_KERNEL = 2,
    int N_ROWS_STRIDE = 2,
    int N_COLS_STRIDE = 2
>
class MaxPoolingIndices
{
public:
    typedef Tensor<
        double,
        InT::DIM_1,
        (InT::DIM_2 - N_ROWS_KERNEL) / N_ROWS_STRIDE + 1,
        (InT::DIM_3 - N_COLS_KERNEL) / N_COLS_STRIDE + 1
    > OutT;

    typedef Tensor<
        int,
        OutT::DIM_1,
        OutT::DIM_2,
        OutT::DIM_3,
        2  // x, y coordinate
    > IndicesT;

    OutT forward(const InT & src, IndicesT& indices)
    {
        OutT result;
        for(int d1 = 0, s1 = 0; d1 < OutT::DIM_1; ++d1, ++s1) {
            for(int d2 = 0; d2 < OutT::DIM_2; ++d2) {
                const int s2_begin = d2 * N_ROWS_STRIDE;
                for(int d3 = 0; d3 < OutT::DIM_3; ++d3) {
                    const int s3_begin = d3 * N_COLS_STRIDE;
                    double& out_value = result.at(d1, d2, d3);
                    int& out_index_row = indices.at(d1, d2, d3, 0);
                    int& out_index_col = indices.at(d1, d2, d3, 1);
                    out_value = src.at(s1, s2_begin, s3_begin);
                    out_index_row = s2_begin;
                    out_index_col = s3_begin;
                    for (int s2 = s2_begin; s2 < s2_begin + N_ROWS_KERNEL; ++s2) {
                        for (int s3 = s3_begin; s3 < s3_begin + N_COLS_KERNEL; ++s3) {
                            const double current_value = src.at(s1, s2, s3);
                            if (out_value < current_value) {
                                out_value = current_value;
                                out_index_row = s2;
                                out_index_col = s3;
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
};