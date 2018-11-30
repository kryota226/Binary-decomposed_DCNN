#pragma once
#include "layers/convolution.hpp"
#include "layers/fast_convolution.hpp"
#include "layers/fast_fully_connected.hpp"
#include "layers/flatten.hpp"
#include "layers/fully_connected.hpp"
#include "layers/linear.hpp"
#include "layers/max_pooling.hpp"
#include "layers/relu.hpp"
#include "layers/softmax.hpp"
#include "utils/tensor.hpp"


class Network
{
public:
    typedef Tensor<double, 3, 224, 224> InT;
    typedef Tensor<double, 1000> OutT;

    Convolution<Tensor<double,   3, 224, 224>, 64, 3, 3, 1, 1, 1, 1> conv1_1;
    //Convolution<Tensor<double,  64, 224, 224>, 64, 3, 3, 1, 1, 1, 1> conv1_2;
    //Convolution<Tensor<double,  64, 112, 112>, 128, 3, 3, 1, 1, 1, 1> conv2_1;
    //Convolution<Tensor<double, 128, 112, 112>, 128, 3, 3, 1, 1, 1, 1> conv2_2;
    //Convolution<Tensor<double, 128,  56,  56>, 256, 3, 3, 1, 1, 1, 1> conv3_1;
    //Convolution<Tensor<double, 256,  56,  56>, 256, 3, 3, 1, 1, 1, 1> conv3_2;
    //Convolution<Tensor<double, 256,  56,  56>, 256, 3, 3, 1, 1, 1, 1> conv3_3;
    //Convolution<Tensor<double, 256,  28,  28>, 512, 3, 3, 1, 1, 1, 1> conv4_1;
    //Convolution<Tensor<double, 512,  28,  28>, 512, 3, 3, 1, 1, 1, 1> conv4_2;
    //Convolution<Tensor<double, 512,  28,  28>, 512, 3, 3, 1, 1, 1, 1> conv4_3;
    //Convolution<Tensor<double, 512,  14,  14>, 512, 3, 3, 1, 1, 1, 1> conv5_1;
    //Convolution<Tensor<double, 512,  14,  14>, 512, 3, 3, 1, 1, 1, 1> conv5_2;
    //Convolution<Tensor<double, 512,  14,  14>, 512, 3, 3, 1, 1, 1, 1> conv5_3;

    //FullyConnected<Tensor<double, 25088>, 4096> fc6;
    //FullyConnected<Tensor<double,  4096>, 4096> fc7;
    //FullyConnected<Tensor<double,  4096>, 1000> fc8;

   // FastConvolution<Tensor<double,   3, 224, 224>,  64, 3, 3, 1, 1, 1, 1,  6, 6> conv1_1;
    FastConvolution<Tensor<double,  64, 224, 224>,  64, 3, 3, 1, 1, 1, 1,  6, 6> conv1_2;
    FastConvolution<Tensor<double,  64, 112, 112>, 128, 3, 3, 1, 1, 1, 1,  6, 6> conv2_1;
    FastConvolution<Tensor<double, 128, 112, 112>, 128, 3, 3, 1, 1, 1, 1,  6, 6> conv2_2;
    FastConvolution<Tensor<double, 128,  56,  56>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv3_1;
    FastConvolution<Tensor<double, 256,  56,  56>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv3_2;
    FastConvolution<Tensor<double, 256,  56,  56>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv3_3;
    FastConvolution<Tensor<double, 256,  28,  28>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv4_1;
    FastConvolution<Tensor<double, 512,  28,  28>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv4_2;
    FastConvolution<Tensor<double, 512,  28,  28>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv4_3;
    FastConvolution<Tensor<double, 512,  14,  14>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_1;
    FastConvolution<Tensor<double, 512,  14,  14>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_2;
    FastConvolution<Tensor<double, 512,  14,  14>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_3;

    FastFullyConnected<Tensor<double, 25088>, 4096,  6, 6> fc6;
    FastFullyConnected<Tensor<double,  4096>, 4096,  6, 6> fc7;
    FastFullyConnected<Tensor<double,  4096>, 1000,  6, 6> fc8;


    ReLU<Tensor<double,  64, 224, 224> > relu1_1;
    ReLU<Tensor<double,  64, 224, 224> > relu1_2;
    ReLU<Tensor<double, 128, 112, 112> > relu2_1;
    ReLU<Tensor<double, 128, 112, 112> > relu2_2;
    ReLU<Tensor<double, 256, 56, 56> > relu3_1;
    ReLU<Tensor<double, 256, 56, 56> > relu3_2;
    ReLU<Tensor<double, 256, 56, 56> > relu3_3;
    ReLU<Tensor<double, 512, 28, 28> > relu4_1;
    ReLU<Tensor<double, 512, 28, 28> > relu4_2;
    ReLU<Tensor<double, 512, 28, 28> > relu4_3;
    ReLU<Tensor<double, 512, 14, 14> > relu5_1;
    ReLU<Tensor<double, 512, 14, 14> > relu5_2;
    ReLU<Tensor<double, 512, 14, 14> > relu5_3;
    ReLU<Tensor<double, 4096> > relu6;
    ReLU<Tensor<double, 4096> > relu7;

    SoftMax<Tensor<double, 1000> > softmax;

    MaxPooling<Tensor<double,  64, 224, 224>, 1, 2, 2> max_pooling1_2;
    MaxPooling<Tensor<double, 128, 112, 112>, 1, 2, 2> max_pooling2_2;
    MaxPooling<Tensor<double, 256,  56,  56>, 1, 2, 2> max_pooling3_3;
    MaxPooling<Tensor<double, 512,  28,  28>, 1, 2, 2> max_pooling4_3;
    MaxPooling<Tensor<double, 512,  14,  14>, 1, 2, 2> max_pooling5_3;

    Flatten<Tensor<double, 512, 7, 7> > flatten;

    Network(const std::string & param_folder);

    OutT forward(InT & src)
    {
        OutT dst = src
        >> conv1_1
        >> relu1_1
        >> conv1_2
        >> relu1_2
        >> max_pooling1_2
        >> conv2_1
        >> relu2_1
        >> conv2_2
        >> relu2_2
        >> max_pooling2_2
        >> conv3_1
        >> relu3_1
        >> conv3_2
        >> relu3_2
        >> conv3_3
        >> relu3_3
        >> max_pooling3_3
        >> conv4_1
        >> relu4_1
        >> conv4_2
        >> relu4_2
        >> conv4_3
        >> relu4_3
        >> max_pooling4_3
        >> conv5_1
        >> relu5_1
        >> conv5_2
        >> relu5_2
        >> conv5_3
        >> relu5_3
        >> max_pooling5_3
        >> flatten
        >> fc6
        >> relu6
        >> fc7
        >> relu7
        >> fc8
        >> softmax;
        return dst;
    }

private:
    std::string param_path;
    std::string origin_param_path;
    std::string decomposed_param_path;
};
