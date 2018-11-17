#pragma once
#include <functional>
#include "utils/tensor.hpp"


template <typename T, int P=0>
class Eltwise
{
public:
    typedef T InT;
    typedef T OutT;

    enum {
        SUM,
    };

	/*
    OutT forward(const InT & src1, const InT & src2)
    {
        if(P == SUM) {
            return src1 + src2;
        }
        else {
            std::cerr << "Non supported process: {}" << std::endl;
            std::exit(-1);
        }
    }
	*/
	OutT operator () (const InT & src1, const InT & src2)
	{
		if (P == SUM) {
			std::vector<double> result(OutT::SIZE);
			std::transform(
				src1.data.begin(), src1.data.end(), src2.data.begin(), result.begin(),
				std::plus<double>()
			);
			return OutT(result);
		}
		else {
			std::cerr << "Non supported process: {}" << std::endl;
			std::exit(-1);
		}
	}

private:
    double multiplier_;
    double accumulator_;
};