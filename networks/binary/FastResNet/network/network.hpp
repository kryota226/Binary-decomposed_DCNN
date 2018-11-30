#pragma once
#include <string>
#include "layers/average_pooling.hpp"
#include "layers/batch_normalization.hpp"
#include "layers/convolution.hpp"
#include "layers/eltwise.hpp"
#include "layers/fast_convolution.hpp"
#include "layers/fast_fully_connected.hpp"
#include "layers/flatten.hpp"
#include "layers/fully_connected.hpp"
#include "layers/max_pooling.hpp"
#include "layers/relu.hpp"
#include "layers/softmax.hpp"
#include "utils/tensor.hpp"


class Network
{
public:
    typedef Tensor<double, 3, 224, 224> InT;
    typedef Tensor<double, 1000> OutT;

    // Convolution layers

	/*
    Convolution<Tensor<double, 3, 224, 224>, 64, 7, 7, 2, 2, 3, 3> conv1;

    Convolution<Tensor<double,  64, 56, 56>, 256, 1, 1, 1, 1, 0, 0> res2a_branch1;
    Convolution<Tensor<double,  64, 56, 56>,  64, 1, 1, 1, 1, 0, 0> res2a_branch2a;
    Convolution<Tensor<double,  64, 56, 56>,  64, 3, 3, 1, 1, 1, 1> res2a_branch2b;
    Convolution<Tensor<double,  64, 56, 56>, 256, 1, 1, 1, 1, 0, 0> res2a_branch2c;
    Convolution<Tensor<double, 256, 56, 56>,  64, 1, 1, 1, 1, 0, 0> res2b_branch2a;
    Convolution<Tensor<double,  64, 56, 56>,  64, 3, 3, 1, 1, 1, 1> res2b_branch2b;
    Convolution<Tensor<double,  64, 56, 56>, 256, 1, 1, 1, 1, 0, 0> res2b_branch2c;
    Convolution<Tensor<double, 256, 56, 56>,  64, 1, 1, 1, 1, 0, 0> res2c_branch2a;
    Convolution<Tensor<double,  64, 56, 56>,  64, 3, 3, 1, 1, 1, 1> res2c_branch2b;
    Convolution<Tensor<double,  64, 56, 56>, 256, 1, 1, 1, 1, 0, 0> res2c_branch2c;

    Convolution<Tensor<double, 256, 56, 56>, 512, 1, 1, 2, 2, 0, 0> res3a_branch1;
    Convolution<Tensor<double, 256, 56, 56>, 128, 1, 1, 2, 2, 0, 0> res3a_branch2a;
    Convolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1> res3a_branch2b;
    Convolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0> res3a_branch2c;
    Convolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0> res3b1_branch2a;
    Convolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1> res3b1_branch2b;
    Convolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0> res3b1_branch2c;
    Convolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0> res3b2_branch2a;
    Convolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1> res3b2_branch2b;
    Convolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0> res3b2_branch2c;
    Convolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0> res3b3_branch2a;
    Convolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1> res3b3_branch2b;
    Convolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0> res3b3_branch2c;
    Convolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0> res3b4_branch2a;
    Convolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1> res3b4_branch2b;
    Convolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0> res3b4_branch2c;
    Convolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0> res3b5_branch2a;
    Convolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1> res3b5_branch2b;
    Convolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0> res3b5_branch2c;
    Convolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0> res3b6_branch2a;
    Convolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1> res3b6_branch2b;
    Convolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0> res3b6_branch2c;
    Convolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0> res3b7_branch2a;
    Convolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1> res3b7_branch2b;
    Convolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0> res3b7_branch2c;

    Convolution<Tensor<double,  512, 28, 28>, 1024, 1, 1, 2, 2, 0, 0> res4a_branch1;
    Convolution<Tensor<double,  512, 28, 28>,  256, 1, 1, 2, 2, 0, 0> res4a_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4a_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4a_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b1_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b1_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b1_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b2_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b2_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b2_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b3_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b3_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b3_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b4_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b4_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b4_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b5_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b5_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b5_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b6_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b6_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b6_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b7_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b7_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b7_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b8_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b8_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b8_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b9_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b9_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b9_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b10_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b10_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b10_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b11_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b11_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b11_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b12_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b12_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b12_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b13_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b13_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b13_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b14_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b14_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b14_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b15_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b15_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b15_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b16_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b16_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b16_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b17_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b17_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b17_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b18_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b18_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b18_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b19_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b19_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b19_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b20_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b20_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b20_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b21_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b21_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b21_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b22_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b22_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b22_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b23_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b23_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b23_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b24_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b24_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b24_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b25_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b25_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b25_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b26_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b26_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b26_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b27_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b27_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b27_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b28_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b28_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b28_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b29_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b29_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b29_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b30_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b30_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b30_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b31_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b31_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b31_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b32_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b32_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b32_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b33_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b33_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b33_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b34_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b34_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b34_branch2c;
    Convolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0> res4b35_branch2a;
    Convolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1> res4b35_branch2b;
    Convolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0> res4b35_branch2c;

    Convolution<Tensor<double, 1024, 14, 14>, 2048, 1, 1, 2, 2, 0, 0> res5a_branch1;
    Convolution<Tensor<double, 1024, 14, 14>,  512, 1, 1, 2, 2, 0, 0> res5a_branch2a;
    Convolution<Tensor<double,  512,  7,  7>,  512, 3, 3, 1, 1, 1, 1> res5a_branch2b;
    Convolution<Tensor<double,  512,  7,  7>, 2048, 1, 1, 1, 1, 0, 0> res5a_branch2c;
    Convolution<Tensor<double, 2048,  7,  7>,  512, 1, 1, 1, 1, 0, 0> res5b_branch2a;
    Convolution<Tensor<double,  512,  7,  7>,  512, 3, 3, 1, 1, 1, 1> res5b_branch2b;
    Convolution<Tensor<double,  512,  7,  7>, 2048, 1, 1, 1, 1, 0, 0> res5b_branch2c;
    Convolution<Tensor<double, 2048,  7,  7>,  512, 1, 1, 1, 1, 0, 0> res5c_branch2a;
    Convolution<Tensor<double,  512,  7,  7>,  512, 3, 3, 1, 1, 1, 1> res5c_branch2b;
    Convolution<Tensor<double,  512,  7,  7>, 2048, 1, 1, 1, 1, 0, 0> res5c_branch2c;
	*/

    // Fast Convolution layers
    FastConvolution<Tensor<double, 3, 224, 224>, 64, 7, 7, 2, 2, 3, 3,  6, 6> conv1;

    FastConvolution<Tensor<double,  64, 56, 56>, 256, 1, 1, 1, 1, 0, 0,  6, 6> res2a_branch1;
    FastConvolution<Tensor<double,  64, 56, 56>,  64, 1, 1, 1, 1, 0, 0,  6, 6> res2a_branch2a;
    FastConvolution<Tensor<double,  64, 56, 56>,  64, 3, 3, 1, 1, 1, 1,  6, 6> res2a_branch2b;
    FastConvolution<Tensor<double,  64, 56, 56>, 256, 1, 1, 1, 1, 0, 0,  6, 6> res2a_branch2c;
    FastConvolution<Tensor<double, 256, 56, 56>,  64, 1, 1, 1, 1, 0, 0,  6, 6> res2b_branch2a;
    FastConvolution<Tensor<double,  64, 56, 56>,  64, 3, 3, 1, 1, 1, 1,  6, 6> res2b_branch2b;
    FastConvolution<Tensor<double,  64, 56, 56>, 256, 1, 1, 1, 1, 0, 0,  6, 6> res2b_branch2c;
    FastConvolution<Tensor<double, 256, 56, 56>,  64, 1, 1, 1, 1, 0, 0,  6, 6> res2c_branch2a;
    FastConvolution<Tensor<double,  64, 56, 56>,  64, 3, 3, 1, 1, 1, 1,  6, 6> res2c_branch2b;
    FastConvolution<Tensor<double,  64, 56, 56>, 256, 1, 1, 1, 1, 0, 0,  6, 6> res2c_branch2c;

    FastConvolution<Tensor<double, 256, 56, 56>, 512, 1, 1, 2, 2, 0, 0,  6, 6> res3a_branch1;
    FastConvolution<Tensor<double, 256, 56, 56>, 128, 1, 1, 2, 2, 0, 0,  6, 6> res3a_branch2a;
    FastConvolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1,  6, 6> res3a_branch2b;
    FastConvolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0,  6, 6> res3a_branch2c;
    FastConvolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0,  6, 6> res3b1_branch2a;
    FastConvolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1,  6, 6> res3b1_branch2b;
    FastConvolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0,  6, 6> res3b1_branch2c;
    FastConvolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0,  6, 6> res3b2_branch2a;
    FastConvolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1,  6, 6> res3b2_branch2b;
    FastConvolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0,  6, 6> res3b2_branch2c;
    FastConvolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0,  6, 6> res3b3_branch2a;
    FastConvolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1,  6, 6> res3b3_branch2b;
    FastConvolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0,  6, 6> res3b3_branch2c;
    FastConvolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0,  6, 6> res3b4_branch2a;
    FastConvolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1,  6, 6> res3b4_branch2b;
    FastConvolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0,  6, 6> res3b4_branch2c;
    FastConvolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0,  6, 6> res3b5_branch2a;
    FastConvolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1,  6, 6> res3b5_branch2b;
    FastConvolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0,  6, 6> res3b5_branch2c;
    FastConvolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0,  6, 6> res3b6_branch2a;
    FastConvolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1,  6, 6> res3b6_branch2b;
    FastConvolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0,  6, 6> res3b6_branch2c;
    FastConvolution<Tensor<double, 512, 28, 28>, 128, 1, 1, 1, 1, 0, 0,  6, 6> res3b7_branch2a;
    FastConvolution<Tensor<double, 128, 28, 28>, 128, 3, 3, 1, 1, 1, 1,  6, 6> res3b7_branch2b;
    FastConvolution<Tensor<double, 128, 28, 28>, 512, 1, 1, 1, 1, 0, 0,  6, 6> res3b7_branch2c;

    FastConvolution<Tensor<double,  512, 28, 28>, 1024, 1, 1, 2, 2, 0, 0,  6, 6> res4a_branch1;
    FastConvolution<Tensor<double,  512, 28, 28>,  256, 1, 1, 2, 2, 0, 0,  6, 6> res4a_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4a_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4a_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b1_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b1_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b1_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b2_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b2_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b2_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b3_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b3_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b3_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b4_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b4_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b4_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b5_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b5_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b5_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b6_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b6_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b6_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b7_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b7_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b7_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b8_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b8_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b8_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b9_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b9_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b9_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b10_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b10_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b10_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b11_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b11_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b11_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b12_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b12_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b12_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b13_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b13_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b13_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b14_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b14_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b14_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b15_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b15_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b15_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b16_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b16_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b16_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b17_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b17_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b17_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b18_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b18_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b18_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b19_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b19_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b19_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b20_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b20_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b20_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b21_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b21_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b21_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b22_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b22_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b22_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b23_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b23_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b23_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b24_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b24_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b24_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b25_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b25_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b25_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b26_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b26_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b26_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b27_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b27_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b27_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b28_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b28_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b28_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b29_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b29_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b29_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b30_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b30_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b30_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b31_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b31_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b31_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b32_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b32_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b32_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b33_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b33_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b33_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b34_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b34_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b34_branch2c;
    FastConvolution<Tensor<double, 1024, 14, 14>,  256, 1, 1, 1, 1, 0, 0,  6, 6> res4b35_branch2a;
    FastConvolution<Tensor<double,  256, 14, 14>,  256, 3, 3, 1, 1, 1, 1,  6, 6> res4b35_branch2b;
    FastConvolution<Tensor<double,  256, 14, 14>, 1024, 1, 1, 1, 1, 0, 0,  6, 6> res4b35_branch2c;

    FastConvolution<Tensor<double, 1024, 14, 14>, 2048, 1, 1, 2, 2, 0, 0,  6, 6> res5a_branch1;
    FastConvolution<Tensor<double, 1024, 14, 14>,  512, 1, 1, 2, 2, 0, 0,  6, 6> res5a_branch2a;
    FastConvolution<Tensor<double,  512,  7,  7>,  512, 3, 3, 1, 1, 1, 1,  6, 6> res5a_branch2b;
    FastConvolution<Tensor<double,  512,  7,  7>, 2048, 1, 1, 1, 1, 0, 0,  6, 6> res5a_branch2c;
    FastConvolution<Tensor<double, 2048,  7,  7>,  512, 1, 1, 1, 1, 0, 0,  6, 6> res5b_branch2a;
    FastConvolution<Tensor<double,  512,  7,  7>,  512, 3, 3, 1, 1, 1, 1,  6, 6> res5b_branch2b;
    FastConvolution<Tensor<double,  512,  7,  7>, 2048, 1, 1, 1, 1, 0, 0,  6, 6> res5b_branch2c;
    FastConvolution<Tensor<double, 2048,  7,  7>,  512, 1, 1, 1, 1, 0, 0,  6, 6> res5c_branch2a;
    FastConvolution<Tensor<double,  512,  7,  7>,  512, 3, 3, 1, 1, 1, 1,  6, 6> res5c_branch2b;
    FastConvolution<Tensor<double,  512,  7,  7>, 2048, 1, 1, 1, 1, 0, 0,  6, 6> res5c_branch2c;


    // Batch normalization layers
    BatchNormalization<Tensor<double, 64, 112, 112> > bn_conv1;

    BatchNormalization<Tensor<double, 256, 56, 56> > bn2a_branch1;
    BatchNormalization<Tensor<double,  64, 56, 56> > bn2a_branch2a;
    BatchNormalization<Tensor<double,  64, 56, 56> > bn2a_branch2b;
    BatchNormalization<Tensor<double, 256, 56, 56> > bn2a_branch2c;
    BatchNormalization<Tensor<double,  64, 56, 56> > bn2b_branch2a;
    BatchNormalization<Tensor<double,  64, 56, 56> > bn2b_branch2b;
    BatchNormalization<Tensor<double, 256, 56, 56> > bn2b_branch2c;
    BatchNormalization<Tensor<double,  64, 56, 56> > bn2c_branch2a;
    BatchNormalization<Tensor<double,  64, 56, 56> > bn2c_branch2b;
    BatchNormalization<Tensor<double, 256, 56, 56> > bn2c_branch2c;

    BatchNormalization<Tensor<double, 512, 28, 28> > bn3a_branch1;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3a_branch2a;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3a_branch2b;
    BatchNormalization<Tensor<double, 512, 28, 28> > bn3a_branch2c;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b1_branch2a;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b1_branch2b;
    BatchNormalization<Tensor<double, 512, 28, 28> > bn3b1_branch2c;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b2_branch2a;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b2_branch2b;
    BatchNormalization<Tensor<double, 512, 28, 28> > bn3b2_branch2c;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b3_branch2a;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b3_branch2b;
    BatchNormalization<Tensor<double, 512, 28, 28> > bn3b3_branch2c;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b4_branch2a;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b4_branch2b;
    BatchNormalization<Tensor<double, 512, 28, 28> > bn3b4_branch2c;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b5_branch2a;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b5_branch2b;
    BatchNormalization<Tensor<double, 512, 28, 28> > bn3b5_branch2c;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b6_branch2a;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b6_branch2b;
    BatchNormalization<Tensor<double, 512, 28, 28> > bn3b6_branch2c;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b7_branch2a;
    BatchNormalization<Tensor<double, 128, 28, 28> > bn3b7_branch2b;
    BatchNormalization<Tensor<double, 512, 28, 28> > bn3b7_branch2c;

    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4a_branch1;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4a_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4a_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4a_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b1_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b1_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b1_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b2_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b2_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b2_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b3_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b3_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b3_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b4_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b4_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b4_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b5_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b5_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b5_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b6_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b6_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b6_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b7_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b7_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b7_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b8_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b8_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b8_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b9_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b9_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b9_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b10_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b10_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b10_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b11_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b11_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b11_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b12_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b12_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b12_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b13_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b13_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b13_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b14_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b14_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b14_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b15_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b15_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b15_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b16_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b16_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b16_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b17_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b17_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b17_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b18_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b18_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b18_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b19_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b19_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b19_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b20_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b20_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b20_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b21_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b21_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b21_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b22_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b22_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b22_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b23_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b23_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b23_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b24_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b24_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b24_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b25_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b25_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b25_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b26_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b26_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b26_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b27_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b27_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b27_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b28_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b28_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b28_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b29_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b29_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b29_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b30_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b30_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b30_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b31_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b31_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b31_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b32_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b32_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b32_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b33_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b33_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b33_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b34_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b34_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b34_branch2c;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b35_branch2a;
    BatchNormalization<Tensor<double,  256, 14, 14> > bn4b35_branch2b;
    BatchNormalization<Tensor<double, 1024, 14, 14> > bn4b35_branch2c;

    BatchNormalization<Tensor<double, 2048, 7, 7> > bn5a_branch1;
    BatchNormalization<Tensor<double,  512, 7, 7> > bn5a_branch2a;
    BatchNormalization<Tensor<double,  512, 7, 7> > bn5a_branch2b;
    BatchNormalization<Tensor<double, 2048, 7, 7> > bn5a_branch2c;
    BatchNormalization<Tensor<double,  512, 7, 7> > bn5b_branch2a;
    BatchNormalization<Tensor<double,  512, 7, 7> > bn5b_branch2b;
    BatchNormalization<Tensor<double, 2048, 7, 7> > bn5b_branch2c;
    BatchNormalization<Tensor<double,  512, 7, 7> > bn5c_branch2a;
    BatchNormalization<Tensor<double,  512, 7, 7> > bn5c_branch2b;
    BatchNormalization<Tensor<double, 2048, 7, 7> > bn5c_branch2c;


    // ReLU layers
    ReLU<Tensor<double,  64, 112, 112> > conv1_relu;

    ReLU<Tensor<double,  64, 56, 56> > res2a_branch2a_relu;
    ReLU<Tensor<double,  64, 56, 56> > res2a_branch2b_relu;
    ReLU<Tensor<double, 256, 56, 56> > res2a_relu;
    ReLU<Tensor<double,  64, 56, 56> > res2b_branch2a_relu;
    ReLU<Tensor<double,  64, 56, 56> > res2b_branch2b_relu;
    ReLU<Tensor<double, 256, 56, 56> > res2b_relu;
    ReLU<Tensor<double,  64, 56, 56> > res2c_branch2a_relu;
    ReLU<Tensor<double,  64, 56, 56> > res2c_branch2b_relu;
    ReLU<Tensor<double, 256, 56, 56> > res2c_relu;

    ReLU<Tensor<double, 128, 28, 28> > res3a_branch2a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3a_branch2b_relu;
    ReLU<Tensor<double, 512, 28, 28> > res3a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b1_branch2a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b1_branch2b_relu;
    ReLU<Tensor<double, 512, 28, 28> > res3b1_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b2_branch2a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b2_branch2b_relu;
    ReLU<Tensor<double, 512, 28, 28> > res3b2_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b3_branch2a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b3_branch2b_relu;
    ReLU<Tensor<double, 512, 28, 28> > res3b3_relu; 
    ReLU<Tensor<double, 128, 28, 28> > res3b4_branch2a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b4_branch2b_relu;
    ReLU<Tensor<double, 512, 28, 28> > res3b4_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b5_branch2a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b5_branch2b_relu;
    ReLU<Tensor<double, 512, 28, 28> > res3b5_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b6_branch2a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b6_branch2b_relu;
    ReLU<Tensor<double, 512, 28, 28> > res3b6_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b7_branch2a_relu;
    ReLU<Tensor<double, 128, 28, 28> > res3b7_branch2b_relu;
    ReLU<Tensor<double, 512, 28, 28> > res3b7_relu;

    ReLU<Tensor<double,  256, 14, 14> > res4a_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4a_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b1_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b1_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b1_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b2_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b2_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b2_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b3_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b3_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b3_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b4_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b4_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b4_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b5_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b5_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b5_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b6_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b6_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b6_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b7_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b7_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b7_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b8_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b8_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b8_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b9_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b9_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b9_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b10_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b10_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b10_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b11_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b11_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b11_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b12_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b12_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b12_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b13_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b13_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b13_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b14_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b14_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b14_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b15_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b15_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b15_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b16_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b16_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b16_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b17_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b17_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b17_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b18_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b18_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b18_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b19_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b19_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b19_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b20_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b20_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b20_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b21_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b21_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b21_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b22_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b22_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b22_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b23_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b23_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b23_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b24_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b24_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b24_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b25_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b25_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b25_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b26_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b26_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b26_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b27_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b27_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b27_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b28_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b28_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b28_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b29_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b29_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b29_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b30_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b30_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b30_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b31_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b31_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b31_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b32_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b32_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b32_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b33_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b33_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b33_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b34_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b34_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b34_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b35_branch2a_relu;
    ReLU<Tensor<double,  256, 14, 14> > res4b35_branch2b_relu;
    ReLU<Tensor<double, 1024, 14, 14> > res4b35_relu;

    ReLU<Tensor<double,  512, 7, 7> > res5a_branch2a_relu;
    ReLU<Tensor<double,  512, 7, 7> > res5a_branch2b_relu;
    ReLU<Tensor<double, 2048, 7, 7> > res5a_relu;
    ReLU<Tensor<double,  512, 7, 7> > res5b_branch2a_relu;
    ReLU<Tensor<double,  512, 7, 7> > res5b_branch2b_relu;
    ReLU<Tensor<double, 2048, 7, 7> > res5b_relu;
    ReLU<Tensor<double,  512, 7, 7> > res5c_branch2a_relu;
    ReLU<Tensor<double,  512, 7, 7> > res5c_branch2b_relu;
    ReLU<Tensor<double, 2048, 7, 7> > res5c_relu;


    // Eltwise layers
    Eltwise<Tensor<double, 256, 56, 56> > res2a;
    Eltwise<Tensor<double, 256, 56, 56> > res2b;
    Eltwise<Tensor<double, 256, 56, 56> > res2c;

    Eltwise<Tensor<double, 512, 28, 28> > res3a;
    Eltwise<Tensor<double, 512, 28, 28> > res3b1;
    Eltwise<Tensor<double, 512, 28, 28> > res3b2;
    Eltwise<Tensor<double, 512, 28, 28> > res3b3;
    Eltwise<Tensor<double, 512, 28, 28> > res3b4;
    Eltwise<Tensor<double, 512, 28, 28> > res3b5;
    Eltwise<Tensor<double, 512, 28, 28> > res3b6;
    Eltwise<Tensor<double, 512, 28, 28> > res3b7;

    Eltwise<Tensor<double, 1024, 14, 14> > res4a;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b1;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b2;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b3;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b4;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b5;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b6;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b7;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b8;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b9;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b10;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b11;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b12;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b13;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b14;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b15;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b16;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b17;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b18;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b19;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b20;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b21;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b22;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b23;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b24;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b25;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b26;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b27;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b28;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b29;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b30;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b31;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b32;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b33;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b34;
    Eltwise<Tensor<double, 1024, 14, 14> > res4b35;

    Eltwise<Tensor<double, 2048, 7, 7> > res5a;
    Eltwise<Tensor<double, 2048, 7, 7> > res5b;
    Eltwise<Tensor<double, 2048, 7, 7> > res5c;

    // Pooling layers
    MaxPooling<Tensor<double, 64, 112, 112>, 1, 3, 3, 2, 2> pool1;
    AveragePooling<Tensor<double, 2048, 7, 7>, 7, 7, 1, 1> pool5;

    Flatten<Tensor<double, 2048, 1, 1> > flatten;
    //FullyConnected<Tensor<double, 2048>, 1000> fc1000;
    FastFullyConnected<Tensor<double, 2048>, 1000,  6, 6> fc1000;
    SoftMax<Tensor<double, 1000> > prob;

    Network(const std::string & param_folder);

    OutT forward(InT & src)
    {
        // block1
        Tensor<double, 64, 56, 56> _pool1 = src >> conv1 >> bn_conv1 >> conv1_relu >> pool1;


        // block2
        Tensor<double, 256, 56, 56> _res2a_branch1 = _pool1
            >> res2a_branch1 >> bn2a_branch1;
        Tensor<double, 256, 56, 56> _res2a_branch2 = _pool1
            >> res2a_branch2a >> bn2a_branch2a >>
            res2a_branch2a_relu
            >> res2a_branch2b >> bn2a_branch2b >> res2a_branch2b_relu
            >> res2a_branch2c >> bn2a_branch2c;
        Tensor<double, 256, 56, 56> _res2a = res2a(_res2a_branch1, _res2a_branch2) >> res2a_relu;

        Tensor<double, 256, 56, 56> _res2b_branch2 = _res2a
            >> res2b_branch2a >> bn2b_branch2a >> res2b_branch2a_relu
            >> res2b_branch2b >> bn2b_branch2b >> res2b_branch2b_relu
            >> res2b_branch2c >> bn2b_branch2c;
        Tensor<double, 256, 56, 56> _res2b = res2b(_res2a, _res2b_branch2) >> res2b_relu;

        Tensor<double, 256, 56, 56> _res2c_branch2 = _res2b
            >> res2c_branch2a >> bn2c_branch2a >> res2c_branch2a_relu
            >> res2c_branch2b >> bn2c_branch2b >> res2c_branch2b_relu
            >> res2c_branch2c >> bn2c_branch2c;
        Tensor<double, 256, 56, 56> _res2c = res2c(_res2b, _res2c_branch2) >> res2c_relu;


        // block3
        Tensor<double, 512, 28, 28> _res3a_branch1 = _res2c
            >> res3a_branch1 >> bn3a_branch1;
        Tensor<double, 512, 28, 28> _res3a_branch2 = _res2c
            >> res3a_branch2a >> bn3a_branch2a >> res3a_branch2a_relu
            >> res3a_branch2b >> bn3a_branch2b >> res3a_branch2b_relu
            >> res3a_branch2c >> bn3a_branch2c;
        Tensor<double, 512, 28, 28> _res3a = res3a(_res3a_branch1, _res3a_branch2) >> res3a_relu;

        Tensor<double, 512, 28, 28> _res3b1_branch2 = _res3a
            >> res3b1_branch2a >> bn3b1_branch2a >> res3b1_branch2a_relu
            >> res3b1_branch2b >> bn3b1_branch2b >> res3b1_branch2b_relu
            >> res3b1_branch2c >> bn3b1_branch2c;
        Tensor<double, 512, 28, 28> _res3b1 = res3b1(_res3a, _res3b1_branch2) >> res3b1_relu;

        Tensor<double, 512, 28, 28> _res3b2_branch2 = _res3b1
            >> res3b2_branch2a >> bn3b2_branch2a >> res3b2_branch2a_relu
            >> res3b2_branch2b >> bn3b2_branch2b >> res3b2_branch2b_relu
            >> res3b2_branch2c >> bn3b2_branch2c;
        Tensor<double, 512, 28, 28> _res3b2 = res3b2(_res3b1, _res3b2_branch2) >> res3b2_relu;

        Tensor<double, 512, 28, 28> _res3b3_branch2 = _res3b2
            >> res3b3_branch2a >> bn3b3_branch2a >> res3b3_branch2a_relu
            >> res3b3_branch2b >> bn3b3_branch2b >> res3b3_branch2b_relu
            >> res3b3_branch2c >> bn3b3_branch2c;
        Tensor<double, 512, 28, 28> _res3b3 = res3b3(_res3b2, _res3b3_branch2) >> res3b3_relu;

        Tensor<double, 512, 28, 28> _res3b4_branch2 = _res3b3
            >> res3b4_branch2a >> bn3b4_branch2a >> res3b4_branch2a_relu
            >> res3b4_branch2b >> bn3b4_branch2b >> res3b4_branch2b_relu
            >> res3b4_branch2c >> bn3b4_branch2c;
        Tensor<double, 512, 28, 28> _res3b4 = res3b4(_res3b3, _res3b4_branch2) >> res3b4_relu;

        Tensor<double, 512, 28, 28> _res3b5_branch2 = _res3b4
            >> res3b5_branch2a >> bn3b5_branch2a >> res3b5_branch2a_relu
            >> res3b5_branch2b >> bn3b5_branch2b >> res3b5_branch2b_relu
            >> res3b5_branch2c >> bn3b5_branch2c;
        Tensor<double, 512, 28, 28> _res3b5 = res3b5(_res3b4, _res3b5_branch2) >> res3b5_relu;

        Tensor<double, 512, 28, 28> _res3b6_branch2 = _res3b5
            >> res3b6_branch2a >> bn3b6_branch2a >> res3b6_branch2a_relu
            >> res3b6_branch2b >> bn3b6_branch2b >> res3b6_branch2b_relu
            >> res3b6_branch2c >> bn3b6_branch2c;
        Tensor<double, 512, 28, 28> _res3b6 = res3b6(_res3b5, _res3b6_branch2) >> res3b6_relu;

        Tensor<double, 512, 28, 28> _res3b7_branch2 = _res3b6
            >> res3b7_branch2a >> bn3b7_branch2a >> res3b7_branch2a_relu
            >> res3b7_branch2b >> bn3b7_branch2b >> res3b7_branch2b_relu
            >> res3b7_branch2c >> bn3b7_branch2c;
        Tensor<double, 512, 28, 28> _res3b7 = res3b7(_res3b6, _res3b7_branch2) >> res3b7_relu;

        // block4
        Tensor<double, 1024, 14, 14> _res4a_branch1 = _res3b7 >> res4a_branch1 >> bn4a_branch1;
        Tensor<double, 1024, 14, 14> _res4a_branch2 = _res3b7
            >> res4a_branch2a >> bn4a_branch2a >> res4a_branch2a_relu
            >> res4a_branch2b >> bn4a_branch2b >> res4a_branch2b_relu
            >> res4a_branch2c >> bn4a_branch2c;
        Tensor<double, 1024, 14, 14> _res4a = res4a(_res4a_branch1, _res4a_branch2) >> res4a_relu;

        Tensor<double, 1024, 14, 14> _res4b1_branch2 = _res4a
            >> res4b1_branch2a >> bn4b1_branch2a >> res4b1_branch2a_relu
            >> res4b1_branch2b >> bn4b1_branch2b >> res4b1_branch2b_relu
            >> res4b1_branch2c >> bn4b1_branch2c;
        Tensor<double, 1024, 14, 14> _res4b1 = res4b1(_res4a, _res4b1_branch2) >> res4b1_relu;

        Tensor<double, 1024, 14, 14> _res4b2_branch2 = _res4b1
            >> res4b2_branch2a >> bn4b2_branch2a >> res4b2_branch2a_relu
            >> res4b2_branch2b >> bn4b2_branch2b >> res4b2_branch2b_relu
            >> res4b2_branch2c >> bn4b2_branch2c;
        Tensor<double, 1024, 14, 14> _res4b2 = res4b2(_res4b1, _res4b2_branch2) >> res4b2_relu;

        Tensor<double, 1024, 14, 14> _res4b3_branch2 = _res4b2
            >> res4b3_branch2a >> bn4b3_branch2a >> res4b3_branch2a_relu
            >> res4b3_branch2b >> bn4b3_branch2b >> res4b3_branch2b_relu
            >> res4b3_branch2c >> bn4b3_branch2c;
        Tensor<double, 1024, 14, 14> _res4b3 = res4b3(_res4b2, _res4b3_branch2) >> res4b3_relu;

        Tensor<double, 1024, 14, 14> _res4b4_branch2 = _res4b3
            >> res4b4_branch2a >> bn4b4_branch2a >> res4b4_branch2a_relu
            >> res4b4_branch2b >> bn4b4_branch2b >> res4b4_branch2b_relu
            >> res4b4_branch2c >> bn4b4_branch2c;
        Tensor<double, 1024, 14, 14> _res4b4 = res4b4(_res4b3, _res4b4_branch2) >> res4b4_relu;

        Tensor<double, 1024, 14, 14> _res4b5_branch2 = _res4b4
            >> res4b5_branch2a >> bn4b5_branch2a >> res4b5_branch2a_relu
            >> res4b5_branch2b >> bn4b5_branch2b >> res4b5_branch2b_relu
            >> res4b5_branch2c >> bn4b5_branch2c;
        Tensor<double, 1024, 14, 14> _res4b5 = res4b5(_res4b4, _res4b5_branch2) >> res4b5_relu;

        Tensor<double, 1024, 14, 14> _res4b6_branch2 = _res4b5
            >> res4b6_branch2a >> bn4b6_branch2a >> res4b6_branch2a_relu
            >> res4b6_branch2b >> bn4b6_branch2b >> res4b6_branch2b_relu
            >> res4b6_branch2c >> bn4b6_branch2c;
        Tensor<double, 1024, 14, 14> _res4b6 = res4b6(_res4b5, _res4b6_branch2) >> res4b6_relu;

        Tensor<double, 1024, 14, 14> _res4b7_branch2 = _res4b6
            >> res4b7_branch2a >> bn4b7_branch2a >> res4b7_branch2a_relu
            >> res4b7_branch2b >> bn4b7_branch2b >> res4b7_branch2b_relu
            >> res4b7_branch2c >> bn4b7_branch2c;
        Tensor<double, 1024, 14, 14> _res4b7 = res4b7(_res4b6, _res4b7_branch2) >> res4b7_relu;

        Tensor<double, 1024, 14, 14> _res4b8_branch2 = _res4b7
            >> res4b8_branch2a >> bn4b8_branch2a >> res4b8_branch2a_relu
            >> res4b8_branch2b >> bn4b8_branch2b >> res4b8_branch2b_relu
            >> res4b8_branch2c >> bn4b8_branch2c;
        Tensor<double, 1024, 14, 14> _res4b8 = res4b8(_res4b7, _res4b8_branch2) >> res4b8_relu;

        Tensor<double, 1024, 14, 14> _res4b9_branch2 = _res4b8
            >> res4b9_branch2a >> bn4b9_branch2a >> res4b9_branch2a_relu
            >> res4b9_branch2b >> bn4b9_branch2b >> res4b9_branch2b_relu
            >> res4b9_branch2c >> bn4b9_branch2c;
        Tensor<double, 1024, 14, 14> _res4b9 = res4b9(_res4b8, _res4b9_branch2) >> res4b9_relu;

        Tensor<double, 1024, 14, 14> _res4b10_branch2 = _res4b9
            >> res4b10_branch2a >> bn4b10_branch2a >> res4b10_branch2a_relu
            >> res4b10_branch2b >> bn4b10_branch2b >> res4b10_branch2b_relu
            >> res4b10_branch2c >> bn4b10_branch2c;
        Tensor<double, 1024, 14, 14> _res4b10 = res4b10(_res4b9, _res4b10_branch2) >> res4b10_relu;

        Tensor<double, 1024, 14, 14> _res4b11_branch2 = _res4b10
            >> res4b11_branch2a >> bn4b11_branch2a >> res4b11_branch2a_relu
            >> res4b11_branch2b >> bn4b11_branch2b >> res4b11_branch2b_relu
            >> res4b11_branch2c >> bn4b11_branch2c;
        Tensor<double, 1024, 14, 14> _res4b11 = res4b11(_res4b10, _res4b11_branch2) >> res4b11_relu;

        Tensor<double, 1024, 14, 14> _res4b12_branch2 = _res4b11
            >> res4b12_branch2a >> bn4b12_branch2a >> res4b12_branch2a_relu
            >> res4b12_branch2b >> bn4b12_branch2b >> res4b12_branch2b_relu
            >> res4b12_branch2c >> bn4b12_branch2c;
        Tensor<double, 1024, 14, 14> _res4b12 = res4b12(_res4b11, _res4b12_branch2) >> res4b12_relu;

        Tensor<double, 1024, 14, 14> _res4b13_branch2 = _res4b12
            >> res4b13_branch2a >> bn4b13_branch2a >> res4b13_branch2a_relu
            >> res4b13_branch2b >> bn4b13_branch2b >> res4b13_branch2b_relu
            >> res4b13_branch2c >> bn4b13_branch2c;
        Tensor<double, 1024, 14, 14> _res4b13 = res4b13(_res4b12, _res4b13_branch2) >> res4b13_relu;

        Tensor<double, 1024, 14, 14> _res4b14_branch2 = _res4b13
            >> res4b14_branch2a >> bn4b14_branch2a >> res4b14_branch2a_relu
            >> res4b14_branch2b >> bn4b14_branch2b >> res4b14_branch2b_relu
            >> res4b14_branch2c >> bn4b14_branch2c;
        Tensor<double, 1024, 14, 14> _res4b14 = res4b14(_res4b13, _res4b14_branch2) >> res4b14_relu;

        Tensor<double, 1024, 14, 14> _res4b15_branch2 = _res4b14
            >> res4b15_branch2a >> bn4b15_branch2a >> res4b15_branch2a_relu
            >> res4b15_branch2b >> bn4b15_branch2b >> res4b15_branch2b_relu
            >> res4b15_branch2c >> bn4b15_branch2c;
        Tensor<double, 1024, 14, 14> _res4b15 = res4b15(_res4b14, _res4b15_branch2) >> res4b15_relu;

        Tensor<double, 1024, 14, 14> _res4b16_branch2 = _res4b15
            >> res4b16_branch2a >> bn4b16_branch2a >> res4b16_branch2a_relu
            >> res4b16_branch2b >> bn4b16_branch2b >> res4b16_branch2b_relu
            >> res4b16_branch2c >> bn4b16_branch2c;
        Tensor<double, 1024, 14, 14> _res4b16 = res4b16(_res4b15, _res4b16_branch2) >> res4b16_relu;

        Tensor<double, 1024, 14, 14> _res4b17_branch2 = _res4b16
            >> res4b17_branch2a >> bn4b17_branch2a >> res4b17_branch2a_relu
            >> res4b17_branch2b >> bn4b17_branch2b >> res4b17_branch2b_relu
            >> res4b17_branch2c >> bn4b17_branch2c;
        Tensor<double, 1024, 14, 14> _res4b17 = res4b17(_res4b16, _res4b17_branch2) >> res4b17_relu;

        Tensor<double, 1024, 14, 14> _res4b18_branch2 = _res4b17
            >> res4b18_branch2a >> bn4b18_branch2a >> res4b18_branch2a_relu
            >> res4b18_branch2b >> bn4b18_branch2b >> res4b18_branch2b_relu
            >> res4b18_branch2c >> bn4b18_branch2c;
        Tensor<double, 1024, 14, 14> _res4b18 = res4b18(_res4b17, _res4b18_branch2) >> res4b18_relu;

        Tensor<double, 1024, 14, 14> _res4b19_branch2 = _res4b18
            >> res4b19_branch2a >> bn4b19_branch2a >> res4b19_branch2a_relu
            >> res4b19_branch2b >> bn4b19_branch2b >> res4b19_branch2b_relu
            >> res4b19_branch2c >> bn4b19_branch2c;
        Tensor<double, 1024, 14, 14> _res4b19 = res4b19(_res4b18, _res4b19_branch2) >> res4b19_relu;

        Tensor<double, 1024, 14, 14> _res4b20_branch2 = _res4b19
            >> res4b20_branch2a >> bn4b20_branch2a >> res4b20_branch2a_relu
            >> res4b20_branch2b >> bn4b20_branch2b >> res4b20_branch2b_relu
            >> res4b20_branch2c >> bn4b20_branch2c;
        Tensor<double, 1024, 14, 14> _res4b20 = res4b20(_res4b19, _res4b20_branch2) >> res4b20_relu;

        Tensor<double, 1024, 14, 14> _res4b21_branch2 = _res4b20
            >> res4b21_branch2a >> bn4b21_branch2a >> res4b21_branch2a_relu
            >> res4b21_branch2b >> bn4b21_branch2b >> res4b21_branch2b_relu
            >> res4b21_branch2c >> bn4b21_branch2c;
        Tensor<double, 1024, 14, 14> _res4b21 = res4b21(_res4b20, _res4b21_branch2) >> res4b21_relu;

        Tensor<double, 1024, 14, 14> _res4b22_branch2 = _res4b21
            >> res4b22_branch2a >> bn4b22_branch2a >> res4b22_branch2a_relu
            >> res4b22_branch2b >> bn4b22_branch2b >> res4b22_branch2b_relu
            >> res4b22_branch2c >> bn4b22_branch2c;
        Tensor<double, 1024, 14, 14> _res4b22 = res4b22(_res4b21, _res4b22_branch2) >> res4b22_relu;

        Tensor<double, 1024, 14, 14> _res4b23_branch2 = _res4b22
            >> res4b23_branch2a >> bn4b23_branch2a >> res4b23_branch2a_relu
            >> res4b23_branch2b >> bn4b23_branch2b >> res4b23_branch2b_relu
            >> res4b23_branch2c >> bn4b23_branch2c;
        Tensor<double, 1024, 14, 14> _res4b23 = res4b23(_res4b22, _res4b23_branch2) >> res4b23_relu;

        Tensor<double, 1024, 14, 14> _res4b24_branch2 = _res4b23
            >> res4b24_branch2a >> bn4b24_branch2a >> res4b24_branch2a_relu
            >> res4b24_branch2b >> bn4b24_branch2b >> res4b24_branch2b_relu
            >> res4b24_branch2c >> bn4b24_branch2c;
        Tensor<double, 1024, 14, 14> _res4b24 = res4b24(_res4b23, _res4b24_branch2) >> res4b24_relu;

        Tensor<double, 1024, 14, 14> _res4b25_branch2 = _res4b24
            >> res4b25_branch2a >> bn4b25_branch2a >> res4b25_branch2a_relu
            >> res4b25_branch2b >> bn4b25_branch2b >> res4b25_branch2b_relu
            >> res4b25_branch2c >> bn4b25_branch2c;
        Tensor<double, 1024, 14, 14> _res4b25 = res4b25(_res4b24, _res4b25_branch2) >> res4b25_relu;

        Tensor<double, 1024, 14, 14> _res4b26_branch2 = _res4b25
            >> res4b26_branch2a >> bn4b26_branch2a >> res4b26_branch2a_relu
            >> res4b26_branch2b >> bn4b26_branch2b >> res4b26_branch2b_relu
            >> res4b26_branch2c >> bn4b26_branch2c;
        Tensor<double, 1024, 14, 14> _res4b26 = res4b26(_res4b25, _res4b26_branch2) >> res4b26_relu;

        Tensor<double, 1024, 14, 14> _res4b27_branch2 = _res4b26
            >> res4b27_branch2a >> bn4b27_branch2a >> res4b27_branch2a_relu
            >> res4b27_branch2b >> bn4b27_branch2b >> res4b27_branch2b_relu
            >> res4b27_branch2c >> bn4b27_branch2c;
        Tensor<double, 1024, 14, 14> _res4b27 = res4b27(_res4b26, _res4b27_branch2) >> res4b27_relu;

        Tensor<double, 1024, 14, 14> _res4b28_branch2 = _res4b27
            >> res4b28_branch2a >> bn4b28_branch2a >> res4b28_branch2a_relu
            >> res4b28_branch2b >> bn4b28_branch2b >> res4b28_branch2b_relu
            >> res4b28_branch2c >> bn4b28_branch2c;
        Tensor<double, 1024, 14, 14> _res4b28 = res4b28(_res4b27, _res4b28_branch2) >> res4b28_relu;

        Tensor<double, 1024, 14, 14> _res4b29_branch2 = _res4b28
            >> res4b29_branch2a >> bn4b29_branch2a >> res4b29_branch2a_relu
            >> res4b29_branch2b >> bn4b29_branch2b >> res4b29_branch2b_relu
            >> res4b29_branch2c >> bn4b29_branch2c;
        Tensor<double, 1024, 14, 14> _res4b29 = res4b29(_res4b28, _res4b29_branch2) >> res4b29_relu;

        Tensor<double, 1024, 14, 14> _res4b30_branch2 = _res4b29
            >> res4b30_branch2a >> bn4b30_branch2a >> res4b30_branch2a_relu
            >> res4b30_branch2b >> bn4b30_branch2b >> res4b30_branch2b_relu
            >> res4b30_branch2c >> bn4b30_branch2c;
        Tensor<double, 1024, 14, 14> _res4b30 = res4b30(_res4b29, _res4b30_branch2) >> res4b30_relu;

        Tensor<double, 1024, 14, 14> _res4b31_branch2 = _res4b30
            >> res4b31_branch2a >> bn4b31_branch2a >> res4b31_branch2a_relu
            >> res4b31_branch2b >> bn4b31_branch2b >> res4b31_branch2b_relu
            >> res4b31_branch2c >> bn4b31_branch2c;
        Tensor<double, 1024, 14, 14> _res4b31 = res4b31(_res4b30, _res4b31_branch2) >> res4b31_relu;

        Tensor<double, 1024, 14, 14> _res4b32_branch2 = _res4b31
            >> res4b32_branch2a >> bn4b32_branch2a >> res4b32_branch2a_relu
            >> res4b32_branch2b >> bn4b32_branch2b >> res4b32_branch2b_relu
            >> res4b32_branch2c >> bn4b32_branch2c;
        Tensor<double, 1024, 14, 14> _res4b32 = res4b32(_res4b31, _res4b32_branch2) >> res4b32_relu;

        Tensor<double, 1024, 14, 14> _res4b33_branch2 = _res4b32
            >> res4b33_branch2a >> bn4b33_branch2a >> res4b33_branch2a_relu
            >> res4b33_branch2b >> bn4b33_branch2b >> res4b33_branch2b_relu
            >> res4b33_branch2c >> bn4b33_branch2c;
        Tensor<double, 1024, 14, 14> _res4b33 = res4b33(_res4b32, _res4b33_branch2) >> res4b33_relu;

        Tensor<double, 1024, 14, 14> _res4b34_branch2 = _res4b33
            >> res4b34_branch2a >> bn4b34_branch2a >> res4b34_branch2a_relu
            >> res4b34_branch2b >> bn4b34_branch2b >> res4b34_branch2b_relu
            >> res4b34_branch2c >> bn4b34_branch2c;
        Tensor<double, 1024, 14, 14> _res4b34 = res4b34(_res4b33, _res4b34_branch2) >> res4b34_relu;

        Tensor<double, 1024, 14, 14> _res4b35_branch2 = _res4b34
            >> res4b35_branch2a >> bn4b35_branch2a >> res4b35_branch2a_relu
            >> res4b35_branch2b >> bn4b35_branch2b >> res4b35_branch2b_relu
            >> res4b35_branch2c >> bn4b35_branch2c;
        Tensor<double, 1024, 14, 14> _res4b35 = res4b35(_res4b34, _res4b35_branch2) >> res4b35_relu;


        // block5
        Tensor<double, 2048, 7, 7> _res5a_branch1 = _res4b35
            >> res5a_branch1 >> bn5a_branch1;
        Tensor<double, 2048, 7, 7> _res5a_branch2 = _res4b35
            >> res5a_branch2a >> bn5a_branch2a >> res5a_branch2a_relu
            >> res5a_branch2b >> bn5a_branch2b >> res5a_branch2b_relu
            >> res5a_branch2c >> bn5a_branch2c;
        Tensor<double, 2048, 7, 7> _res5a = res5a(_res5a_branch1, _res5a_branch2) >> res5a_relu;

        Tensor<double, 2048, 7, 7> _res5b_branch2 = _res5a
            >> res5b_branch2a >> bn5b_branch2a >> res5b_branch2a_relu
            >> res5b_branch2b >> bn5b_branch2b >> res5b_branch2b_relu
            >> res5b_branch2c >> bn5b_branch2c;
        Tensor<double, 2048, 7, 7> _res5b = res5b(_res5a, _res5b_branch2) >> res5b_relu;

        Tensor<double, 2048, 7, 7> _res5c_branch2 = _res5b
            >> res5c_branch2a >> bn5c_branch2a >> res5c_branch2a_relu
            >> res5c_branch2b >> bn5c_branch2b >> res5c_branch2b_relu
            >> res5c_branch2c >> bn5c_branch2c;
        Tensor<double, 2048, 7, 7> _res5c = res5c(_res5b, _res5c_branch2) >> res5c_relu;

        return _res5c >> pool5 >> flatten >> fc1000 >> prob;
    }
};
