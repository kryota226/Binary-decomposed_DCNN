#pragma once
#include <string>
#include "layers/convolution.hpp"
#include "layers/fast_convolution.hpp"
#include "layers/fast_fully_connected.hpp"
#include "layers/flatten.hpp"
#include "layers/fully_connected.hpp"
#include "layers/linear.hpp"
#include "layers/lrn.hpp"
#include "layers/max_pooling.hpp"
#include "layers/relu.hpp"
#include "layers/softmax.hpp"
#include "utils/tensor.hpp"


class Network
{
public:
    typedef Tensor<double, 3, 227, 227> InT;
    typedef Tensor<double, 1000> OutT;

    //Convolution<Tensor<double, 3, 227, 227>, 96, 11, 11, 4, 4> conv1;
    FastConvolution<Tensor<double, 3, 227, 227>, 96, 11, 11, 4, 4, 0, 0,  6, 6> conv1;
    ReLU<Tensor<double, 96, 55, 55> > relu1;
    LocalResponseNormalization<Tensor<double, 96, 55, 55> > lrn1;
    MaxPooling<Tensor<double, 96, 55, 55>, 1, 3, 3, 2, 2> max_pooling1;

    //Convolution<Tensor<double, 96, 27, 27>, 256, 5, 5, 1, 1, 2, 2> conv2;
    FastConvolution<Tensor<double, 96, 27, 27>, 256, 5, 5, 1, 1, 2, 2,  6, 6> conv2;
    ReLU<Tensor<double, 256, 27, 27> > relu2;
    LocalResponseNormalization<Tensor<double, 256, 27, 27> > lrn2;
    MaxPooling<Tensor<double, 256, 27, 27>, 1, 3, 3, 2, 2> max_pooling2;

    //Convolution<Tensor<double, 256, 13, 13>, 384, 3, 3, 1, 1, 1, 1> conv3;
    FastConvolution<Tensor<double, 256, 13, 13>, 384, 3, 3, 1, 1, 1, 1,  6, 6> conv3;
    ReLU<Tensor<double, 384, 13, 13> > relu3;

    //Convolution<Tensor<double, 384, 13, 13>, 384, 3, 3, 1, 1, 1, 1> conv4;
    FastConvolution<Tensor<double, 384, 13, 13>, 384, 3, 3, 1, 1, 1, 1,  6, 6> conv4;
    ReLU<Tensor<double, 384, 13, 13> > relu4;

    //Convolution<Tensor<double, 384, 13, 13>, 256, 3, 3, 1, 1, 1, 1> conv5;
    FastConvolution<Tensor<double, 384, 13, 13>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv5;
    ReLU<Tensor<double, 256, 13, 13> > relu5;
    MaxPooling<Tensor<double, 256, 13, 13>, 1, 3, 3, 2, 2> max_pooling5;

    Flatten<Tensor<double, 256, 6, 6> > flatten;

    //FullyConnected<Tensor<double, 9216>, 4096> fc6;
    FastFullyConnected<Tensor<double, 9216>, 4096,  6, 6> fc6;
    ReLU<Tensor<double, 4096> > relu6;

    //FullyConnected<Tensor<double, 4096>, 4096> fc7;
    FastFullyConnected<Tensor<double, 4096>, 4096,  6, 6> fc7;
    ReLU<Tensor<double, 4096> > relu7;

    //FullyConnected<Tensor<double, 4096>, 1000> fc8;
    FastFullyConnected<Tensor<double, 4096>, 1000,  6, 6> fc8;
    SoftMax<Tensor<double, 1000> > softmax;

    Network(void);

    OutT forward(InT & src)
    {
        OutT dst = src
            >> conv1
            >> relu1
            >> lrn1
            >> max_pooling1
            >> conv2
            >> relu2
            >> lrn2
            >> max_pooling2
            >> conv3
            >> relu3
            >> conv4
            >> relu4
            >> conv5
            >> relu5
            >> max_pooling5
            >> flatten
            >> fc6
            >> relu6
            >> fc7
            >> relu7
            >> fc8
            >> softmax;
        return dst;
    }
};
