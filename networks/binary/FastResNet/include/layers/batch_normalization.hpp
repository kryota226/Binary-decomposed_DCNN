#pragma once
#include <string>
#include <iomanip>
#include "utils/io.hpp"
#include "utils/tensor.hpp"


template <typename T>
class BatchNormalization
{
public:
    typedef T InT;
    typedef T OutT;
    typedef Tensor<double, InT::DIM_1> MeanT;
    typedef Tensor<double, InT::DIM_1> VarT;
    typedef Tensor<double, InT::DIM_1> ScaleT;
    typedef Tensor<double, InT::DIM_1> BiasT;

    BatchNormalization(
        const std::string & mean_file,
        const std::string & var_file,
        const std::string & scale_file,
        const std::string & bias_file
    ) :
        mean(io::load<double>(mean_file)),
        var(io::load<double>(var_file)),
        scale(io::load<double>(scale_file)),
        bias(io::load<double>(bias_file))
    {}

    OutT forward(const InT & src)
    {
        OutT result;
        for(int d1 = 0, s1 = 0; d1 < OutT::DIM_1; ++d1, ++s1) {
            const double _mean = mean.at(d1);
			const double _stddev = std::sqrtf(static_cast<float>(var.at(d1) + 1e-5));
			const double _scale = scale.at(d1);
			const double _bias = bias.at(d1);
			for (int d2 = 0, s2 = 0; d2 < OutT::DIM_2; ++d2, ++s2) {
                for(int d3 = 0, s3 = 0; d3 < OutT::DIM_3; ++d3, ++s3) {
				    const double x_hat = (src.at(s1, s2, s3) - _mean) / _stddev;
					result.at(d1, d2, d3) = _scale * x_hat + _bias;
                }
            }
        }
        return result;
    }

private:
    const MeanT mean;
    const VarT var;
    const ScaleT scale;
    const BiasT bias;
};