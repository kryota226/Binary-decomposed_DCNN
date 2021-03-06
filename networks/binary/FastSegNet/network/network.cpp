#include "network.hpp"


SegNet::SegNet(const std::string & param_dir) :
    conv1_1(param_dir + "/conv1_1_W.npy", param_dir + "/conv1_1_b.npy"),
    conv1_2(param_dir + "/conv1_2_W.npy", param_dir + "/conv1_2_b.npy",
            param_dir + "/decomposition/conv1_2/basis_6/M.npy", param_dir + "/decomposition/conv1_2/basis_6/c.npy"),
    conv2_1(param_dir + "/conv2_1_W.npy", param_dir + "/conv2_1_b.npy",
            param_dir + "/decomposition/conv2_1/basis_6/M.npy", param_dir + "/decomposition/conv2_1/basis_6/c.npy"),
    conv2_2(param_dir + "/conv2_2_W.npy", param_dir + "/conv2_2_b.npy",
            param_dir + "/decomposition/conv2_2/basis_6/M.npy", param_dir + "/decomposition/conv2_2/basis_6/c.npy"),
    conv3_1(param_dir + "/conv3_1_W.npy", param_dir + "/conv3_1_b.npy",
            param_dir + "/decomposition/conv3_1/basis_6/M.npy", param_dir + "/decomposition/conv3_1/basis_6/c.npy"),
    conv3_2(param_dir + "/conv3_2_W.npy", param_dir + "/conv3_2_b.npy",
            param_dir + "/decomposition/conv3_2/basis_6/M.npy", param_dir + "/decomposition/conv3_2/basis_6/c.npy"),
    conv3_3(param_dir + "/conv3_3_W.npy", param_dir + "/conv3_3_b.npy",
            param_dir + "/decomposition/conv3_3/basis_6/M.npy", param_dir + "/decomposition/conv3_3/basis_6/c.npy"),
    conv4_1(param_dir + "/conv4_1_W.npy", param_dir + "/conv4_1_b.npy",
            param_dir + "/decomposition/conv4_1/basis_6/M.npy", param_dir + "/decomposition/conv4_1/basis_6/c.npy"),
    conv4_2(param_dir + "/conv4_2_W.npy", param_dir + "/conv4_2_b.npy",
            param_dir + "/decomposition/conv4_2/basis_6/M.npy", param_dir + "/decomposition/conv4_2/basis_6/c.npy"),
    conv4_3(param_dir + "/conv4_3_W.npy", param_dir + "/conv4_3_b.npy",
            param_dir + "/decomposition/conv4_3/basis_6/M.npy", param_dir + "/decomposition/conv4_3/basis_6/c.npy"),
    conv5_1(param_dir + "/conv5_1_W.npy", param_dir + "/conv5_1_b.npy",
            param_dir + "/decomposition/conv5_1/basis_6/M.npy", param_dir + "/decomposition/conv5_1/basis_6/c.npy"),
    conv5_2(param_dir + "/conv5_2_W.npy", param_dir + "/conv5_2_b.npy",
            param_dir + "/decomposition/conv5_2/basis_6/M.npy", param_dir + "/decomposition/conv5_2/basis_6/c.npy"),
    conv5_3(param_dir + "/conv5_3_W.npy", param_dir + "/conv5_3_b.npy",
            param_dir + "/decomposition/conv5_3/basis_6/M.npy", param_dir + "/decomposition/conv5_3/basis_6/c.npy"),

    conv5_3_D(param_dir + "/conv5_3_D_W.npy", param_dir + "/conv5_3_D_b.npy",
              param_dir + "/decomposition/conv5_3_D/basis_6/M.npy", param_dir + "/decomposition/conv5_3_D/basis_6/c.npy"),
    conv5_2_D(param_dir + "/conv5_2_D_W.npy", param_dir + "/conv5_2_D_b.npy",
              param_dir + "/decomposition/conv5_2_D/basis_6/M.npy", param_dir + "/decomposition/conv5_2_D/basis_6/c.npy"),
    conv5_1_D(param_dir + "/conv5_1_D_W.npy", param_dir + "/conv5_1_D_b.npy",
              param_dir + "/decomposition/conv5_1_D/basis_6/M.npy", param_dir + "/decomposition/conv5_1_D/basis_6/c.npy"),
    conv4_3_D(param_dir + "/conv4_3_D_W.npy", param_dir + "/conv4_3_D_b.npy",
              param_dir + "/decomposition/conv4_3_D/basis_6/M.npy", param_dir + "/decomposition/conv4_3_D/basis_6/c.npy"),
    conv4_2_D(param_dir + "/conv4_2_D_W.npy", param_dir + "/conv4_2_D_b.npy",
              param_dir + "/decomposition/conv4_2_D/basis_6/M.npy", param_dir + "/decomposition/conv4_2_D/basis_6/c.npy"),
    conv4_1_D(param_dir + "/conv4_1_D_W.npy", param_dir + "/conv4_1_D_b.npy",
              param_dir + "/decomposition/conv4_1_D/basis_6/M.npy", param_dir + "/decomposition/conv4_1_D/basis_6/c.npy"),
    conv3_3_D(param_dir + "/conv3_3_D_W.npy", param_dir + "/conv3_3_D_b.npy",
              param_dir + "/decomposition/conv3_3_D/basis_6/M.npy", param_dir + "/decomposition/conv3_3_D/basis_6/c.npy"),
    conv3_2_D(param_dir + "/conv3_2_D_W.npy", param_dir + "/conv3_2_D_b.npy",
              param_dir + "/decomposition/conv3_2_D/basis_6/M.npy", param_dir + "/decomposition/conv3_2_D/basis_6/c.npy"),
    conv3_1_D(param_dir + "/conv3_1_D_W.npy", param_dir + "/conv3_1_D_b.npy",
              param_dir + "/decomposition/conv3_1_D/basis_6/M.npy", param_dir + "/decomposition/conv3_1_D/basis_6/c.npy"),
    conv2_2_D(param_dir + "/conv2_2_D_W.npy", param_dir + "/conv2_2_D_b.npy",
              param_dir + "/decomposition/conv2_2_D/basis_6/M.npy", param_dir + "/decomposition/conv2_2_D/basis_6/c.npy"),
    conv2_1_D(param_dir + "/conv2_1_D_W.npy", param_dir + "/conv2_1_D_b.npy",
              param_dir + "/decomposition/conv2_1_D/basis_6/M.npy", param_dir + "/decomposition/conv2_1_D/basis_6/c.npy"),
    conv1_2_D(param_dir + "/conv1_2_D_W.npy", param_dir + "/conv1_2_D_b.npy",
              param_dir + "/decomposition/conv1_2_D/basis_6/M.npy", param_dir + "/decomposition/conv1_2_D/basis_6/c.npy"),
    conv1_1_D(param_dir + "/conv1_1_D_W.npy", param_dir + "/conv1_1_D_b.npy",
              param_dir + "/decomposition/conv1_1_D/basis_6/M.npy", param_dir + "/decomposition/conv1_1_D/basis_6/c.npy"),

    conv1_1_bn(param_dir + "/conv1_1_bn_gamma.npy", param_dir + "/conv1_1_bn_shift.npy"),
    conv1_2_bn(param_dir + "/conv1_2_bn_gamma.npy", param_dir + "/conv1_2_bn_shift.npy"),
    conv2_1_bn(param_dir + "/conv2_1_bn_gamma.npy", param_dir + "/conv2_1_bn_shift.npy"),
    conv2_2_bn(param_dir + "/conv2_2_bn_gamma.npy", param_dir + "/conv2_2_bn_shift.npy"),
    conv3_1_bn(param_dir + "/conv3_1_bn_gamma.npy", param_dir + "/conv3_1_bn_shift.npy"),
    conv3_2_bn(param_dir + "/conv3_2_bn_gamma.npy", param_dir + "/conv3_2_bn_shift.npy"),
    conv3_3_bn(param_dir + "/conv3_3_bn_gamma.npy", param_dir + "/conv3_3_bn_shift.npy"),
    conv4_1_bn(param_dir + "/conv4_1_bn_gamma.npy", param_dir + "/conv4_1_bn_shift.npy"),
    conv4_2_bn(param_dir + "/conv4_2_bn_gamma.npy", param_dir + "/conv4_2_bn_shift.npy"),
    conv4_3_bn(param_dir + "/conv4_3_bn_gamma.npy", param_dir + "/conv4_3_bn_shift.npy"),
    conv5_1_bn(param_dir + "/conv5_1_bn_gamma.npy", param_dir + "/conv5_1_bn_shift.npy"),
    conv5_2_bn(param_dir + "/conv5_2_bn_gamma.npy", param_dir + "/conv5_2_bn_shift.npy"),
    conv5_3_bn(param_dir + "/conv5_3_bn_gamma.npy", param_dir + "/conv5_3_bn_shift.npy"),

    conv5_3_D_bn(param_dir + "/conv5_3_D_bn_gamma.npy", param_dir + "/conv5_3_D_bn_shift.npy"),
    conv5_2_D_bn(param_dir + "/conv5_2_D_bn_gamma.npy", param_dir + "/conv5_2_D_bn_shift.npy"),
    conv5_1_D_bn(param_dir + "/conv5_1_D_bn_gamma.npy", param_dir + "/conv5_1_D_bn_shift.npy"),
    conv4_3_D_bn(param_dir + "/conv4_3_D_bn_gamma.npy", param_dir + "/conv4_3_D_bn_shift.npy"),
    conv4_2_D_bn(param_dir + "/conv4_2_D_bn_gamma.npy", param_dir + "/conv4_2_D_bn_shift.npy"),
    conv4_1_D_bn(param_dir + "/conv4_1_D_bn_gamma.npy", param_dir + "/conv4_1_D_bn_shift.npy"),
    conv3_3_D_bn(param_dir + "/conv3_3_D_bn_gamma.npy", param_dir + "/conv3_3_D_bn_shift.npy"),
    conv3_2_D_bn(param_dir + "/conv3_2_D_bn_gamma.npy", param_dir + "/conv3_2_D_bn_shift.npy"),
    conv3_1_D_bn(param_dir + "/conv3_1_D_bn_gamma.npy", param_dir + "/conv3_1_D_bn_shift.npy"),
    conv2_2_D_bn(param_dir + "/conv2_2_D_bn_gamma.npy", param_dir + "/conv2_2_D_bn_shift.npy"),
    conv2_1_D_bn(param_dir + "/conv2_1_D_bn_gamma.npy", param_dir + "/conv2_1_D_bn_shift.npy"),
    conv1_2_D_bn(param_dir + "/conv1_2_D_bn_gamma.npy", param_dir + "/conv1_2_D_bn_shift.npy")
{}