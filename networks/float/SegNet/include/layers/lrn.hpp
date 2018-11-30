#pragma once
//#include "utils/math.hpp"
#include "utils/tensor.hpp"


template <typename T>
class LocalResponseNormalization
{
public:
    typedef T InT;
    typedef T OutT;

    LocalResponseNormalization(
        const int n=5, const int k=2, const double alpha=1e-4, const double beta=0.75
    ) : local_size(n), k_(k), alpha_(alpha), beta_(beta)
    {
        if(local_size % 2 == 0) {
            std::cerr << "LRN only supported even value for local_size." << std::endl;
            std::exit(-1);
        }
    }

    OutT forward(const InT & src)
    {
        InT square_src = src.square();
        const int half_n = local_size / 2;

        OutT result;
        for(int d1 = 0; d1 < OutT::DIM_1; ++d1) {
            const int s1 = d1;
            const int begin_map = std::max(0, d1 - half_n);
            const int end_map = std::min(static_cast<int>(InT::DIM_1), d1 + half_n + 1);
            for(int d2 = 0; d2 < OutT::DIM_2; ++d2) {
                const int s2 = d2;
                for(int d3 = 0; d3 < OutT::DIM_3; ++d3) {
                    const int s3 = d3;
                    double sum_part = 0.0;
                    for(int ss1 = begin_map; ss1 < end_map; ++ss1) {
                        sum_part += square_src.at(ss1, s2, s3);
                    }
                    const double scale = std::pow(sum_part * (alpha_ / local_size) + 1, -beta_);
                    result.at(d1, d2, d3) = scale * src.at(s1, s2, s3);
                }
            }
        }
        return result;
    }

private:
    const int k_;
    const int local_size;
    const double alpha_;
    const double beta_;
};