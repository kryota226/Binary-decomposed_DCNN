#pragma once
#include "layers/batch_normalization.hpp"
#include "layers/convolution.hpp"
#include "layers/fast_convolution.hpp"
#include "layers/linear.hpp"
#include "layers/max_pooling_indices.hpp"
#include "layers/relu.hpp"
#include "layers/softmax.hpp"
#include "layers/unpooling.hpp"
#include "utils/tensor.hpp"


class SegNet
{
public:
    typedef Tensor<double, 3, 360, 480> InT;
    typedef Tensor<double, 12, 360, 480> OutT;
    typedef Tensor<double, 64, 180, 240> Pool1;
    typedef Tensor<double, 128, 90, 120> Pool2;
    typedef Tensor<double, 256, 45, 60> Pool3;
    typedef Tensor<double, 512, 22, 30> Pool4;
    typedef Tensor<double, 512, 11, 15> Pool5;
    typedef Tensor<double, 512, 23, 30> Pool4_D;
    typedef Tensor<double, 256, 45, 60> Pool3_D;
    typedef Tensor<double, 128, 90, 120> Pool2_D;
    typedef Tensor<double, 64, 180, 240> Pool1_D;
    typedef Tensor<int, 64, 180, 240, 2> Pool1_Indices;
    typedef Tensor<int, 128, 90, 120, 2> Pool2_Indices;
    typedef Tensor<int, 256, 45, 60, 2> Pool3_Indices;
    typedef Tensor<int, 512, 22, 30, 2> Pool4_Indices;
    typedef Tensor<int, 512, 11, 15, 2> Pool5_Indices;

    Convolution<Tensor<double, 3, 360, 480>, 64, 3, 3, 1, 1, 1, 1> conv1_1;
    BatchNormalization<Tensor<double, 64, 360, 480> > conv1_1_bn;
    ReLU<Tensor<double, 64, 360, 480> > relu1_1;
    //Convolution<Tensor<double, 64, 360, 480>, 64, 3, 3, 1, 1, 1, 1> conv1_2;
    FastConvolution<Tensor<double, 64, 360, 480>, 64, 3, 3, 1, 1, 1, 1,  6, 6> conv1_2;
    BatchNormalization<Tensor<double, 64, 360, 480> > conv1_2_bn;
    ReLU<Tensor<double, 64, 360, 480> > relu1_2;
    MaxPoolingIndices<Tensor<double, 64, 360, 480>, 2, 2> pool1;

    //Convolution<Tensor<double, 64, 180, 240>, 128, 3, 3, 1, 1, 1, 1> conv2_1;
    FastConvolution<Tensor<double, 64, 180, 240>, 128, 3, 3, 1, 1, 1, 1,  6, 6> conv2_1;
    BatchNormalization<Tensor<double, 128, 180, 240> > conv2_1_bn;
    ReLU<Tensor<double, 128, 180, 240> > relu2_1;
    //Convolution<Tensor<double, 128, 180, 240>, 128, 3, 3, 1, 1, 1, 1> conv2_2;
    FastConvolution<Tensor<double, 128, 180, 240>, 128, 3, 3, 1, 1, 1, 1,  6, 6> conv2_2;
    BatchNormalization<Tensor<double, 128, 180, 240> > conv2_2_bn;
    ReLU<Tensor<double, 128, 180, 240> > relu2_2;
    MaxPoolingIndices<Tensor<double, 128, 180, 240>, 2, 2> pool2;

    //Convolution<Tensor<double, 128, 90, 120>, 256, 3, 3, 1, 1, 1, 1> conv3_1;
    FastConvolution<Tensor<double, 128, 90, 120>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv3_1;
    BatchNormalization<Tensor<double, 256, 90, 120> > conv3_1_bn;
    ReLU<Tensor<double, 256, 90, 120> > relu3_1;
    //Convolution<Tensor<double, 256, 90, 120>, 256, 3, 3, 1, 1, 1, 1> conv3_2;
    FastConvolution<Tensor<double, 256, 90, 120>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv3_2;
    BatchNormalization<Tensor<double, 256, 90, 120> > conv3_2_bn;
    ReLU<Tensor<double, 256, 90, 120> > relu3_2;
    //Convolution<Tensor<double, 256, 90, 120>, 256, 3, 3, 1, 1, 1, 1> conv3_3;
    FastConvolution<Tensor<double, 256, 90, 120>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv3_3;
    BatchNormalization<Tensor<double, 256, 90, 120> > conv3_3_bn;
    ReLU<Tensor<double, 256, 90, 120> > relu3_3;
    MaxPoolingIndices<Tensor<double, 256, 90, 120>, 2, 2> pool3;

    //Convolution<Tensor<double, 256, 45, 60>, 512, 3, 3, 1, 1, 1, 1> conv4_1;
    FastConvolution<Tensor<double, 256, 45, 60>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv4_1;
    BatchNormalization<Tensor<double, 512, 45, 60> > conv4_1_bn;
    ReLU<Tensor<double, 512, 45, 60> > relu4_1;
    //Convolution<Tensor<double, 512, 45, 60>, 512, 3, 3, 1, 1, 1, 1> conv4_2;
    FastConvolution<Tensor<double, 512, 45, 60>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv4_2;
    BatchNormalization<Tensor<double, 512, 45, 60> > conv4_2_bn;
    ReLU<Tensor<double, 512, 45, 60> > relu4_2;
    //Convolution<Tensor<double, 512, 45, 60>, 512, 3, 3, 1, 1, 1, 1> conv4_3;
    FastConvolution<Tensor<double, 512, 45, 60>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv4_3;
    BatchNormalization<Tensor<double, 512, 45, 60> > conv4_3_bn;
    ReLU<Tensor<double, 512, 45, 60> > relu4_3;
    MaxPoolingIndices<Tensor<double, 512, 45, 60>, 2, 2> pool4;

    //Convolution<Tensor<double, 512, 22, 30>, 512, 3, 3, 1, 1, 1, 1> conv5_1;
    FastConvolution<Tensor<double, 512, 22, 30>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_1;
    BatchNormalization<Tensor<double, 512, 22, 30> > conv5_1_bn;
    ReLU<Tensor<double, 512, 22, 30> > relu5_1;
    //Convolution<Tensor<double, 512, 22, 30>, 512, 3, 3, 1, 1, 1, 1> conv5_2;
    FastConvolution<Tensor<double, 512, 22, 30>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_2;
    BatchNormalization<Tensor<double, 512, 22, 30> > conv5_2_bn;
    ReLU<Tensor<double, 512, 22, 30> > relu5_2;
    //Convolution<Tensor<double, 512, 22, 30>, 512, 3, 3, 1, 1, 1, 1> conv5_3;
    FastConvolution<Tensor<double, 512, 22, 30>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_3;
    BatchNormalization<Tensor<double, 512, 22, 30> > conv5_3_bn;
    ReLU<Tensor<double, 512, 22, 30> > relu5_3;
    MaxPoolingIndices<Tensor<double, 512, 22, 30>, 2, 2> pool5;

    Unpooling<Tensor<double, 512, 11, 15>, Pool5_Indices, 2, 23, 30> upsample5;
    //Convolution<Tensor<double, 512, 23, 30>, 512, 3, 3, 1, 1, 1, 1> conv5_3_D;
    FastConvolution<Tensor<double, 512, 23, 30>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_3_D;
    BatchNormalization<Tensor<double, 512, 23, 30> > conv5_3_D_bn;
    ReLU<Tensor<double, 512, 23, 30> > relu5_3_D;
    //Convolution<Tensor<double, 512, 23, 30>, 512, 3, 3, 1, 1, 1, 1> conv5_2_D;
    FastConvolution<Tensor<double, 512, 23, 30>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_2_D;
    BatchNormalization<Tensor<double, 512, 23, 30> > conv5_2_D_bn;
    ReLU<Tensor<double, 512, 23, 30> > relu5_2_D;
    //Convolution<Tensor<double, 512, 23, 30>, 512, 3, 3, 1, 1, 1, 1> conv5_1_D;
    FastConvolution<Tensor<double, 512, 23, 30>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv5_1_D;
    BatchNormalization<Tensor<double, 512, 23, 30> > conv5_1_D_bn;
    ReLU<Tensor<double, 512, 23, 30> > relu5_1_D;

    Unpooling<Tensor<double, 512, 23, 30>, Pool4_Indices, 2, 45, 60> upsample4;
    //Convolution<Tensor<double, 512, 45, 60>, 512, 3, 3, 1, 1, 1, 1> conv4_3_D;
    FastConvolution<Tensor<double, 512, 45, 60>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv4_3_D;
    BatchNormalization<Tensor<double, 512, 45, 60> > conv4_3_D_bn;
    ReLU<Tensor<double, 512, 45, 60> > relu4_3_D;
    //Convolution<Tensor<double, 512, 45, 60>, 512, 3, 3, 1, 1, 1, 1> conv4_2_D;
    FastConvolution<Tensor<double, 512, 45, 60>, 512, 3, 3, 1, 1, 1, 1,  6, 6> conv4_2_D;
    BatchNormalization<Tensor<double, 512, 45, 60> > conv4_2_D_bn;
    ReLU<Tensor<double, 512, 45, 60> > relu4_2_D;
    //Convolution<Tensor<double, 512, 45, 60>, 256, 3, 3, 1, 1, 1, 1> conv4_1_D;
    FastConvolution<Tensor<double, 512, 45, 60>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv4_1_D;
    BatchNormalization<Tensor<double, 256, 45, 60> > conv4_1_D_bn;
    ReLU<Tensor<double, 256, 45, 60> > relu4_1_D;

    Unpooling<Tensor<double, 256, 45, 60>, Pool3_Indices, 2> upsample3;
    //Convolution<Tensor<double, 256, 90, 120>, 256, 3, 3, 1, 1, 1, 1> conv3_3_D;
    FastConvolution<Tensor<double, 256, 90, 120>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv3_3_D;
    BatchNormalization<Tensor<double, 256, 90, 120> > conv3_3_D_bn;
    ReLU<Tensor<double, 256, 90, 120> > relu3_3_D;
    //Convolution<Tensor<double, 256, 90, 120>, 256, 3, 3, 1, 1, 1, 1> conv3_2_D;
    FastConvolution<Tensor<double, 256, 90, 120>, 256, 3, 3, 1, 1, 1, 1,  6, 6> conv3_2_D;
    BatchNormalization<Tensor<double, 256, 90, 120> > conv3_2_D_bn;
    ReLU<Tensor<double, 256, 90, 120> > relu3_2_D;
    //Convolution<Tensor<double, 256, 90, 120>, 128, 3, 3, 1, 1, 1, 1> conv3_1_D;
    FastConvolution<Tensor<double, 256, 90, 120>, 128, 3, 3, 1, 1, 1, 1,  6, 6> conv3_1_D;
    BatchNormalization<Tensor<double, 128, 90, 120> > conv3_1_D_bn;
    ReLU<Tensor<double, 128, 90, 120> > relu3_1_D;

    Unpooling<Tensor<double, 128, 90, 120>, Pool2_Indices, 2> upsample2;
    //Convolution<Tensor<double, 128, 180, 240>, 128, 3, 3, 1, 1, 1, 1> conv2_2_D;
    FastConvolution<Tensor<double, 128, 180, 240>, 128, 3, 3, 1, 1, 1, 1,  6, 6> conv2_2_D;
    BatchNormalization<Tensor<double, 128, 180, 240> > conv2_2_D_bn;
    ReLU<Tensor<double, 128, 180, 240> > relu2_2_D;
    //Convolution<Tensor<double, 128, 180, 240>, 64, 3, 3, 1, 1, 1, 1> conv2_1_D;
    FastConvolution<Tensor<double, 128, 180, 240>, 64, 3, 3, 1, 1, 1, 1,  6, 6> conv2_1_D;
    BatchNormalization<Tensor<double, 64, 180, 240> > conv2_1_D_bn;
    ReLU<Tensor<double, 64, 180, 240> > relu2_1_D;

    Unpooling<Tensor<double, 64, 180, 240>, Pool1_Indices, 2> upsample1;
    //Convolution<Tensor<double, 64, 360, 480>, 64, 3, 3, 1, 1, 1, 1> conv1_2_D;
    FastConvolution<Tensor<double, 64, 360, 480>, 64, 3, 3, 1, 1, 1, 1,  6, 6> conv1_2_D;
    BatchNormalization<Tensor<double, 64, 360, 480> > conv1_2_D_bn;
    ReLU<Tensor<double, 64, 360, 480> > relu1_2_D;
    //Convolution<Tensor<double, 64, 360, 480>, 12, 3, 3, 1, 1, 1, 1> conv1_1_D;
    FastConvolution<Tensor<double, 64, 360, 480>, 12, 3, 3, 1, 1, 1, 1,  6, 6> conv1_1_D;

    SoftMax<Tensor<double, 12, 360, 480> > softmax;

    SegNet(const std::string & param_folder);

    OutT forward(InT & src)
    {
        Pool1_Indices pool1_indices;
        Pool2_Indices pool2_indices;
        Pool3_Indices pool3_indices;
        Pool4_Indices pool4_indices;
        Pool5_Indices pool5_indices;

        Pool1 pool1_ = pool1.forward(src    >> conv1_1 >> conv1_1_bn >> relu1_1 >> conv1_2 >> conv1_2_bn >> relu1_2, pool1_indices);
        Pool2 pool2_ = pool2.forward(pool1_ >> conv2_1 >> conv2_1_bn >> relu2_1 >> conv2_2 >> conv2_2_bn >> relu2_2, pool2_indices);
        Pool3 pool3_ = pool3.forward(pool2_ >> conv3_1 >> conv3_1_bn >> relu3_1 >> conv3_2 >> conv3_2_bn >> relu3_2 >> conv3_3 >> conv3_3_bn >> relu3_3, pool3_indices);
        Pool4 pool4_ = pool4.forward(pool3_ >> conv4_1 >> conv4_1_bn >> relu4_1 >> conv4_2 >> conv4_2_bn >> relu4_2 >> conv4_3 >> conv4_3_bn >> relu4_3, pool4_indices);
        Pool5 pool5_ = pool5.forward(pool4_ >> conv5_1 >> conv5_1_bn >> relu5_1 >> conv5_2 >> conv5_2_bn >> relu5_2 >> conv5_3 >> conv5_3_bn >> relu5_3, pool5_indices);

        /*
        for (int d1 = 0; d1 < Pool1::DIM_1; ++d1) {
            for (int d2 = 0; d2 < Pool1::DIM_2; ++d2) {
                for (int d3 = 0; d3 < Pool1::DIM_3; ++d3) {
                    std::cout << pool1_indices.at(d1, d2, d3, 0) << ", " << pool1_indices.at(d1, d2, d3, 1) << std::endl;
                    getchar();
                }
            }
        }
        */

        Pool4_D pool4_d = upsample5.forward(pool5_, pool5_indices)  >> conv5_3_D >> conv5_3_D_bn >> relu5_3_D >> conv5_2_D >> conv5_2_D_bn >> relu5_2_D >> conv5_1_D >> conv5_1_D_bn >> relu5_1_D;
        Pool3_D pool3_d = upsample4.forward(pool4_d, pool4_indices) >> conv4_3_D >> conv4_3_D_bn >> relu4_3_D >> conv4_2_D >> conv4_2_D_bn >> relu4_2_D >> conv4_1_D >> conv4_1_D_bn >> relu4_1_D;
        Pool2_D pool2_d = upsample3.forward(pool3_d, pool3_indices) >> conv3_3_D >> conv3_3_D_bn >> relu3_3_D >> conv3_2_D >> conv3_2_D_bn >> relu3_2_D >> conv3_1_D >> conv3_1_D_bn >> relu3_1_D;
        Pool1_D pool1_d = upsample2.forward(pool2_d, pool2_indices)                                           >> conv2_2_D >> conv2_2_D_bn >> relu2_2_D >> conv2_1_D >> conv2_1_D_bn >> relu2_1_D;
        return upsample1.forward(pool1_d, pool1_indices)                                                      >> conv1_2_D >> conv1_2_D_bn >> relu1_2_D >> conv1_1_D                 >> softmax;
    }
};