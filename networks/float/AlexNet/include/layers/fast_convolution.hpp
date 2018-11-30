#pragma once
#include <cstdint>
#include <nmmintrin.h>
#include <string>
#include <vector>
#include "utils/iceil.hpp"
#include "utils/io.hpp"
#include "utils/math.hpp"
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
    int N_COLS_PAD=0,
    int QUANTIZE_BITS=1,
    int N_BASIS=1
>
class FastConvolution
{
public:
    enum {
        IN_SIZE = InT::DIM_1 * N_ROWS_FILTER * N_COLS_FILTER,
        N_BITSET = iceil<IN_SIZE, 64>::value,
    };
    static_assert(0 < N_BASIS, "N_BASIS is too small.");
    static_assert(0 < QUANTIZE_BITS, "QUANTIZE_BITS is too small.");

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

    typedef Tensor<double, N_OUT_MAPS, InT::DIM_1, N_ROWS_FILTER, N_COLS_FILTER> WeightT;
    typedef Tensor<double, N_OUT_MAPS> BiasT;
    typedef Tensor<double, WeightT::DIM_1, N_BASIS> CoeffVectors;
    typedef Tensor<double, QUANTIZE_BITS> RestoreCoeff;
    typedef Tensor<double, WeightT::DIM_1> Offset;
    typedef Tensor<double, OutT::DIM_2, OutT::DIM_3, QUANTIZE_BITS> Norm;
    typedef Tensor<std::uint64_t, OutT::DIM_2, OutT::DIM_3, QUANTIZE_BITS, N_BITSET> BinaryInT;
    typedef Tensor<std::uint64_t, OutT::DIM_1, N_BASIS, N_BITSET> BinaryMatrices;

    FastConvolution(
        const std::string & weight_file, const std::string & bias_file,
        const std::string & M_file, const std::string & c_file
    ) : bias_(io::load<double>(bias_file)), c(io::load<double>(c_file))
    {
        compression(io::load<int>(M_file));
        set_offset(io::load<double>(weight_file));
    }

    void compression(const std::vector<int> & M)
    {
        for(int d1 = 0; d1 < OutT::DIM_1; ++d1) {
            for(int k = 0; k < N_BASIS; ++k) {
                for(int dim = 0; dim < IN_SIZE; ++dim) {
                    const int dim_ui = dim / 64;
                    std::uint64_t & w_ref = M_ui.at(d1, k, dim_ui);
                    w_ref = (w_ref << 1) + (1 == M[(d1 * N_BASIS + k) * IN_SIZE + dim] ? 1 : 0);
                }
            }
        }
    }

    void set_offset(const std::vector<double> & weight)
    {
        for(int w1 = 0; w1 < WeightT::DIM_1; ++w1) {
            const auto begin = weight.begin() + w1 * IN_SIZE;
            const auto end = begin + IN_SIZE;
            offset.at(w1) = std::accumulate(begin, end, 0.0);
        }
    }

    OutT forward(const InT & src)
    {
        timer.start();

        PadT pad_src = (1 <= N_ROWS_PAD || 1 <= N_COLS_PAD)
            ? zero_padding(src) : PadT(src.data);

        // ********** 入力の最大ｰ最小値間を量子化 ********* //
        const double max_val = math::max(pad_src);
        const double min_val = math::min(pad_src);
        const double quantize_level = (max_val - min_val) / (std::pow(2.0, QUANTIZE_BITS) - 1);
        pad_src = (pad_src - min_val) / quantize_level;

        BinaryInT src_ui(0);
        Norm x_norm;

        for(int d2 = 0; d2 < OutT::DIM_2; ++d2) {
            const int s2_begin = d2 * N_ROWS_STRIDE;
            for(int d3 = 0; d3 < OutT::DIM_3; ++d3) {
                const int s3_begin = d3 * N_COLS_STRIDE;
                int dim = 0;
                for(int s1 = 0; s1 < InT::DIM_1; ++s1) {
                    for(int w3 = 0; w3 < WeightT::DIM_3; ++w3) {
                        const int s2 = s2_begin + w3;
                        for(int w4 = 0; w4 < WeightT::DIM_4; ++w4) {
                            const int s3 = s3_begin + w4;
                            const int round_val = static_cast<int>(pad_src.at(s1, s2, s3) + 0.5);
                            // _mm_popcntが使えるように1つのベクトルに同じ指数(2^k)のバイナリデータを置く
                            const int dim_ui = dim++ / 64;
                            for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
                                std::uint64_t & src_ui_ref = src_ui.at(d2, d3, exp, dim_ui);
                                src_ui_ref = (src_ui_ref << 1) + ((round_val >> exp) & 1);
                            }
                        }
                    }
                }
                for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
                    double & accum_norm = x_norm.at(d2, d3, exp);
                    accum_norm = 0.0;
                    for(int dim_ui = 0; dim_ui < N_BITSET; ++dim_ui) {
                        accum_norm += _mm_popcnt_u64(src_ui.at(d2, d3, exp, dim_ui));
                    }
                }
            }
        }

        RestoreCoeff restore_coeff;
        for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
            restore_coeff.at(exp) = quantize_level * std::pow(2.0, exp);
        }

        // ********** 内積計算 ********** //
        OutT result;

        for(int d1 = 0; d1 < OutT::DIM_1; ++d1) {
            const double offset_ = offset.at(d1) * min_val;
            for(int d2 = 0; d2 < OutT::DIM_2; ++d2) {
                for(int d3 = 0; d3 < OutT::DIM_3; ++d3) {
                    double cMx = bias_.at(d1);
                    for(int k = 0; k < N_BASIS; ++k) {
                        double Mx = 0.0;
                        for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
                            double accum_dist = 0.0;
                            for(int dim_ui = 0; dim_ui < N_BITSET; ++dim_ui) {
                                const std::uint64_t & wb_ref = M_ui.at(d1, k, dim_ui);
                                const std::uint64_t & xb_ref = src_ui.at(d2, d3, exp, dim_ui);
                                accum_dist += _mm_popcnt_u64(wb_ref & xb_ref);
                            }
                            Mx += (2.0 * accum_dist - x_norm.at(d2, d3, exp)) * restore_coeff.at(exp);
                        }
                        cMx += c.at(d1, k) * Mx;
                    }
                    result.at(d1, d2, d3) = cMx + offset_;
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

    BiasT bias_;
    BinaryMatrices M_ui;
    CoeffVectors c;
    Offset offset;

    Timer timer;
};