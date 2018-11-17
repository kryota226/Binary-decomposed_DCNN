#pragma once
#include "utils/io.hpp"
#include "utils/tensor.hpp"
#include "utils/timer.hpp"


template <
    typename InT,
    int N_OUT_MAPS,
    int N_ROWS_FILTER,
    int N_COLS_FILTER,
    int N_ROWS_STRIDE=1,
    int N_COLS_STRIDE=1,
    int N_ROWS_PAD=0,
    int N_COLS_PAD=0
>
class Convolution
{
public:
    typedef Tensor<
        double,
        InT::DIM_1,
        InT::DIM_2 + (2 * N_ROWS_PAD),
        InT::DIM_3 + (2 * N_COLS_PAD)
    > PadT;

    typedef Tensor<
        double,
        N_OUT_MAPS,
        (InT::DIM_2 - N_ROWS_FILTER + (2 * N_ROWS_PAD)) / N_ROWS_STRIDE + 1,
        (InT::DIM_3 - N_COLS_FILTER + (2 * N_COLS_PAD)) / N_COLS_STRIDE + 1
    > OutT;

    typedef Tensor<
        double,
        N_OUT_MAPS,
        InT::DIM_1,
        N_ROWS_FILTER,
        N_COLS_FILTER
    > WeightT;

    typedef Tensor<
        double,
        N_OUT_MAPS
    > BiasT;

    Convolution(const WeightT & weight, const BiasT & bias)
        : weight_(weight), bias_(bias)
    {}

    Convolution(const std::string & weight_npy, const std::string & bias_npy)
        : weight_(io::load<double>(weight_npy)), bias_(io::load<double>(bias_npy))
    {}

    OutT forward(const InT & src)
    {
        timer.start();

        enum {
            W3_BEGIN = WeightT::DIM_3 - 1,
            W4_BEGIN = WeightT::DIM_4 - 1,
        };

        PadT pad_src = (1 <= N_ROWS_PAD || 1 <= N_COLS_PAD)
            ? zero_padding(src) : PadT(src.data);

        OutT result;
        for(int d1 = 0; d1 < OutT::DIM_1; ++d1) {
            const int w1 = d1;
            for(int d2 = 0; d2 < OutT::DIM_2; ++d2) {
                const int d2_ = d2 * N_ROWS_STRIDE;
                for (int d3 = 0; d3 < OutT::DIM_3; ++d3) {
                    const int d3_ = d3 * N_COLS_STRIDE;
                    double sumSoFar = bias_.at(d1);
                    for(int w2 = 0; w2 < WeightT::DIM_2; ++w2) {
                        const int s1 = w2;
                        for(int w3 = W3_BEGIN; 0 <= w3; --w3) {
                            const int s2 = d2_ + W3_BEGIN - w3;
                            for(int w4 = W4_BEGIN; 0 <= w4; --w4) {
                                const int s3 = d3_ + W4_BEGIN - w4;
                                const double w_value = weight_.at(w1, w2, w3, w4);
                                const double s_value = pad_src.at(s1, s2, s3);
                                sumSoFar += w_value * s_value;
                            }
                        }
                    }
                    result.at(d1, d2, d3) += sumSoFar;
                }
            }
        }

        timer.stop();
        return result;
    }

    double get_run_time(void)
    {
        return timer.time();
    }

private:
    PadT zero_padding(const InT & src) const
    {
        PadT pad_src;
        for(int ch = 0; ch < InT::DIM_1; ++ch) {
            for(int row = 0; row < InT::DIM_2; ++row) {
                const int pad_row = N_ROWS_PAD + row;
                for(int col = 0; col < InT::DIM_3; ++col) {
                    const int pad_col = N_COLS_PAD + col;
                    pad_src.at(ch, pad_row, pad_col) = src.at(ch, row, col);
                }
            }
        }
        return pad_src;
    }

    WeightT weight_;
    BiasT bias_;
    Timer timer;
};