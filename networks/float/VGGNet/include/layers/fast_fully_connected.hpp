#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "utils/iceil.hpp"
#include "utils/io.hpp"
#include "utils/tensor.hpp"
#include "utils/timer.hpp"


template <typename InT, int N_OUT_UNITS, int QUANTIZE_BITS=1, int N_BASIS=1>
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
        const double max_val = src.max();
        const double min_val = src.min();
        const double quantize_level
            = (max_val - min_val) / (std::pow(2.0, QUANTIZE_BITS) - 1);
        InT src_ = (src - min_val) / quantize_level;
        
        BinaryInT src_ui(0);
        for(int dim = 0; dim < InT::DIM_1; ++dim) {
            const int round_val = static_cast<int>(src_.at(dim) + 0.5);
            // _mm_popcnt が使えるように同じ指数(2^k)のバイナリ値をマップとしてまとめる
            const int dim_ui = dim / 64;
            for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
                src_ui.at(exp, dim_ui) = (src_ui.at(exp, dim_ui) << 1) + ((round_val >> exp) & 1);
            }
        }

        Coeff restore_coeff(quantize_level);
        for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
            restore_coeff.at(exp) *= std::pow(2.0, exp);
        }

        Coeff app_bitc(0);
        for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
            for(int dim_ui = 0; dim_ui < N_BITSET; ++dim_ui) {
                app_bitc.at(exp) += _mm_popcnt_u64(src_ui.at(exp, dim_ui));
            }
        }

        // ********** 内積計算 ********** //
        OutT result;
        for(int d1 = 0; d1 < OutT::DIM_1; ++d1) {
            double cMx = bias_.at(d1);
            for(int basis = 0; basis < N_BASIS; ++basis) {
                double Mx = 0;
                for(int exp = 0; exp < QUANTIZE_BITS; ++exp) {
                    // 同じ復元係数を持つベクトルのハミング距離をとる
                    int accum_dist = 0;
                    for(int dim_ui = 0; dim_ui < N_BITSET; ++ dim_ui) {
                        std::uint64_t Mx_and = src_ui.at(exp, dim_ui) & M_ui.at(d1, basis, dim_ui);
                        accum_dist += static_cast<int>(_mm_popcnt_u64(Mx_and));
                    }
                    Mx += (2.0 * accum_dist - app_bitc.at(exp)) * restore_coeff.at(exp);
                }
                cMx += c.at(d1, basis) * Mx;
            }
            const double offset_ = offset.at(d1) * min_val;
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