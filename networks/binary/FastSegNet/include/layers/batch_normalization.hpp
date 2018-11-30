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
    typedef Tensor<double, InT::DIM_1> ScaleT;
    typedef Tensor<double, InT::DIM_1> ShiftT;

    BatchNormalization(
        const std::string& scale_file,
        const std::string& shift_file
    ) :
        scale(io::load<double>(scale_file)),
        shift(io::load<double>(shift_file))
    {}

    OutT forward(const InT & src)
    {
        OutT result;
        for(int d1 = 0, s1 = 0; d1 < OutT::DIM_1; ++d1, ++s1) {
            const double _scale = scale.at(d1);
            const double _shift = shift.at(d1);
            for (int d2 = 0, s2 = 0; d2 < OutT::DIM_2; ++d2, ++s2) {
                for(int d3 = 0, s3 = 0; d3 < OutT::DIM_3; ++d3, ++s3) {
                    result.at(d1, d2, d3) = _scale * src.at(s1, s2, s3) + _shift;
                }
            }
        }
        return result;
    }

private:
    const ScaleT scale;
    const ShiftT shift;
};