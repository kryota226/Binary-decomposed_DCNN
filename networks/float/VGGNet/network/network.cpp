#include "network.hpp"


Network::Network(const std::string & param_folder) :
    // conv1_1 �͑����Ȃ�Ȃ��̂ŋߎ��v�Z�Ȃ�
    //conv1_1(param_folder + "/conv1_1_weight.npy"                 , param_folder + "/conv1_1_bias.npy",
           // param_folder + "/decomposition/conv1_1/basis_6/M.npy", param_folder + "/decomposition/conv1_1/basis_6/c.npy"),
	conv1_1(param_folder + "/conv1_1_weight.npy", param_folder + "/conv1_1_bias.npy"),
	conv1_2(param_folder + "/conv1_2_weight.npy"                 , param_folder + "/conv1_2_bias.npy"),
    conv2_1(param_folder + "/conv2_1_weight.npy"                 , param_folder + "/conv2_1_bias.npy"),
    conv2_2(param_folder + "/conv2_2_weight.npy"                 , param_folder + "/conv2_2_bias.npy"),
    conv3_1(param_folder + "/conv3_1_weight.npy"                 , param_folder + "/conv3_1_bias.npy"),
    conv3_2(param_folder + "/conv3_2_weight.npy"                 , param_folder + "/conv3_2_bias.npy"),
    conv3_3(param_folder + "/conv3_3_weight.npy"                 , param_folder + "/conv3_3_bias.npy"),
    conv4_1(param_folder + "/conv4_1_weight.npy"                 , param_folder + "/conv4_1_bias.npy"),
    conv4_2(param_folder + "/conv4_2_weight.npy"                 , param_folder + "/conv4_2_bias.npy"),
    conv4_3(param_folder + "/conv4_3_weight.npy"                 , param_folder + "/conv4_3_bias.npy"),
    conv5_1(param_folder + "/conv5_1_weight.npy"                 , param_folder + "/conv5_1_bias.npy"),
    conv5_2(param_folder + "/conv5_2_weight.npy"                 , param_folder + "/conv5_2_bias.npy"),
    conv5_3(param_folder + "/conv5_3_weight.npy"                 , param_folder + "/conv5_3_bias.npy"),
    fc6(param_folder + "/fc6_weight.npy"                 , param_folder + "/fc6_bias.npy"),
    fc7(param_folder + "/fc7_weight.npy"                 , param_folder + "/fc7_bias.npy"),
    fc8(param_folder + "/fc8_weight.npy"                 , param_folder + "/fc8_bias.npy")
{}