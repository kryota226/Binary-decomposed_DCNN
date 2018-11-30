#pragma once
#include <cstdint>
#include <nmmintrin.h>
#include <string>
#include <vector>
#include "utils/iceil.hpp"
#include "utils/io.hpp"
#include "utils/tensor.hpp"
#include "utils/timer.hpp"


template <typename InT, int N_OUT_UNITS, int N_BASIS=1, int QUANTIZE_BITS=1>
class FastFullyConnected
{
public:
    enum {
        N_BITSET = iceil<InT::DIM_1, 64>::value,
    };
    static_assert(1 <= N_BASIS, "");
    static_assert(1 <= QUANTIZE_BITS, "");
    static_assert(InT::DIM_1 == InT::SIZE, "");

    typedef Tensor<double, N_OUT_UNITS> OutT;
    typedef Tensor<double, InT::DIM_1, N_OUT_UNITS> WeightT_T;
    typedef Tensor<double, N_OUT_UNITS, InT::DIM_1> WeightT;
    typedef Tensor<double, N_OUT_UNITS> BiasT;
    typedef Tensor<double, WeightT::DIM_1, N_BASIS> BasisVectors;
    typedef Tensor<double, WeightT::DIM_1> Offset;
    typedef Tensor<double, QUANTIZE_BITS> Norm;
    typedef Tensor<double, QUANTIZE_BITS> Coeff;
    typedef Tensor<std::uint64_t, QUANTIZE_BITS, N_BITSET> BinaryInT;
    typedef Tensor<std::uint64_t, WeightT::DIM_1, N_BASIS, N_BITSET> BinaryMatrices;

    FastFullyConnected(
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
                for(int dim = 0; dim < InT::DIM_1; ++dim) {
                    const int dim_ui = dim / 64;
                    std::uint64_t & w_ref = M_ui.at(d1, k, dim_ui);
                    w_ref = (w_ref << 1) + (1 == M[(d1 * N_BASIS + k) * InT::DIM_1 + dim] ? 1 : 0);
                }
            }
        }
    }

    void set_offset(const std::vector<double> & weight)
    {
        /*
        Quantization sub-layer の近似内積計算は以下で計算できる。
            y = w_approx * x_approx + min(x) * sum(w)
        このときの畳み込みに使う特徴マップ(c*wh*ww)毎に sum(w) を求めている。
        */
        for(int w1 = 0; w1 < WeightT::DIM_1; ++w1) {
            const auto begin = weight.begin() + w1 * InT::DIM_1;
            const auto end = begin + InT::DIM_1;
            offset.at(w1) = std::accumulate(begin, end, 0.0);
        }
    }

    OutT forward(const InT & src)
    {
        timer.start();

        /********** Quantization sub-layer による入力の量子化 **********/
        // 最大-最小値間を量子化
        const double max_val = src.max();
        const double min_val = src.min();
        const double quantize_level
            = (max_val - min_val) / (std::pow(2.0, QUANTIZE_BITS) - 1);
        InT src_ = (src - min_val) / quantize_level;

        BinaryInT src_ui(0);
        for(int dim = 0; dim < InT::DIM_1; ++dim) {
            const int round_val = static_cast<int>(src_.at(dim) + 0.5);
            const int dim_ui = dim / 64;
            for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
                std::uint64_t & src_ref = src_ui.at(exp, dim_ui);
                src_ref = (src_ref << 1) + ((round_val >> exp) & 1);
            }
        }
        // 復元係数を計算
        Coeff restore_coeff(quantize_level);
        for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
            restore_coeff.at(exp) *= std::pow(2.0, exp);
        }

        // |x|を算出 = x の 1 が立っている数
        Norm x_norm(0);
        for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
            double & accum_norm = x_norm.at(exp);
            for(int dim_ui = 0; dim_ui < N_BITSET; ++dim_ui) {
                accum_norm += _mm_popcnt_u64(src_ui.at(exp, dim_ui));
            }
        }

        // ********** 内積計算 ********** //
        OutT result;
        for(int d1 = 0; d1 < OutT::DIM_1; ++d1) {
            const double offset_ = offset.at(d1) * min_val;
            double cMx = bias_.at(d1);
            for(int k = 0; k < N_BASIS; ++k) {
                double Mx = 0;
                for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
                    double accum_dist = 0.0;
                    for(int dim_ui = 0; dim_ui < N_BITSET; ++ dim_ui) {
                        const std::uint64_t & xb_ref = src_ui.at(exp, dim_ui);
                        const std::uint64_t & wb_ref = M_ui.at(d1, k, dim_ui);
                        accum_dist += _mm_popcnt_u64(xb_ref & wb_ref);
                    }
                    Mx += (2.0 * accum_dist - x_norm.at(exp)) * restore_coeff.at(exp);
                }
                cMx += c.at(d1, k) * Mx;
            }
            result.at(d1) = cMx + offset_;
        }

        timer.stop();
        return result;
    }

    double get_run_time(void)
    {
        return timer.time();
    }

private:
    BiasT bias_;
    BinaryMatrices M_ui;
    BasisVectors c;
    Offset offset;

    Timer timer;
};