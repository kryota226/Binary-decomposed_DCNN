#pragma once
#include "utils/io.hpp"
#include "utils/tensor.hpp"
#include "utils/timer.hpp"


template <typename InT, int N_OUT_UNITS>
class FullyConnected
{
public:
    typedef Tensor<double, N_OUT_UNITS> OutT;
    typedef Tensor<double, InT::SIZE, N_OUT_UNITS> WeightT_T;
    typedef Tensor<double, N_OUT_UNITS, InT::SIZE> WeightT;
    typedef Tensor<double, N_OUT_UNITS> BiasT;

    FullyConnected(const WeightT_T & weight, const BiasT & bias)
        : weight_(transpose(weight)), bias_(bias)
    {}

    FullyConnected(const std::string & weight_npy, const std::string & bias_npy)
        : weight_(io::load<double>(weight_npy)), bias_(io::load<double>(bias_npy))
    {}

    OutT forward(const InT & src)
    {
        timer.start();

        OutT result;
#pragma omp parallel for
        for(int d1 = 0; d1 < OutT::DIM_1; ++d1) {
            double sumSoFar = bias_.at(d1);
#pragma omp parallel for
            for(int s1 = 0; s1 < InT::DIM_1; ++s1) {
                sumSoFar += weight_.at(d1, s1) * src.at(s1);
            }
            result.at(d1) += sumSoFar;
        }

        timer.stop();
        return result;
    }

    double get_run_time(void)
    {
        return timer.time();
    }

private:
    WeightT transpose(const WeightT_T & weight)
    {
        WeightT weight__;
        for(int w1 = 0; w1 < WeightT::DIM_1; ++w1) {
            for(int w2 = 0; w2 < WeightT::DIM_2; ++w2) {
                weight__.at(w1, w2) = weight.at(w2, w1);
            }
        }
        return weight__;
    }

    WeightT weight_;
    BiasT bias_;
    Timer timer;
};