#include <cassert>
#include <iomanip>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "network.hpp"
#include "utils/io.hpp"
#include "utils/timer.hpp"


void load_list(
    const std::string & dataset,
    const std::string & basepath,
    std::vector<std::pair<std::string, int> > & path_list
) {
    std::ifstream ifs(dataset);
    while(ifs.peek() != std::ios::traits_type::eof()) {
        std::string filename;
        int label;
        ifs >> filename >> label;
        filename = basepath + (basepath == "" ? "" : "/") + filename;
        path_list.push_back(
            std::make_pair(filename, label)
        );
    }
}

Network::InT load_image(
    const std::string & filename, const int n_channels
) {
    const auto load_color = n_channels == 3
        ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
    cv::Mat image = cv::imread(filename, load_color);
    if(image.empty()) {
        std::cerr << "CANNOT read files" << std::endl;
        std::exit(-1);
    }
    image.convertTo(image, CV_64FC3);
    cv::Mat mean_image(
        image.size(),
        CV_64FC3,
        cv::Scalar(103.939, 116.779, 123.68)
    );
    image -= mean_image;
    cv::resize(image, image, cv::Size(Network::InT::DIM_3, Network::InT::DIM_2));

    std::vector<cv::Mat1d> blocks;
    cv::split(image, blocks);
    std::vector<double> data;
    for(cv::Mat1d block : blocks) {
        data.insert(data.end(), block.begin(), block.end());
    }
    return Network::InT(data);
}


template <int n_orders>
void evaluation(
    const Network::OutT & predict,
    std::vector<int> & evaluated,
    std::vector<double> & score
) {
    assert(n_orders <= Network::OutT::SIZE);
    std::vector<std::pair<double, int>> prob_index;
    for(int i = 0; i < Network::OutT::SIZE; ++i) {
        prob_index.push_back(
            std::make_pair(predict.at(i), i)
        );
    }
    std::partial_sort(
        prob_index.begin(),
        prob_index.begin() + n_orders,
        prob_index.end(),
        std::greater<std::pair<double, int>>()
    );
    score.resize(n_orders);
    evaluated.resize(n_orders);
    for(int i = 0; i < n_orders; ++i) {
        score[i] = prob_index[i].first;
        evaluated[i] = prob_index[i].second;
    }
}


int main(const int argc, const char **argv)
{
    enum {
        ARGV0 = 0,
        ARG_DATASET,
        ARG_BASE_PATH,
        ARG_PARAM_FOLDER,
        ARG_SAVE_FOLDER,
        NUM_ARGS,
    };
    assert(NUM_ARGS <= argc);

    std::vector<std::pair<std::string, int> > path_list;
    load_list(argv[ARG_DATASET], argv[ARG_BASE_PATH], path_list);

    enum { TOP1, TOP5, N_EVALS };
    std::vector<int> tp_counter(N_EVALS, 0);
    Timer timer;
    Network net(argv[ARG_PARAM_FOLDER]);
    for(size_t i = 0; i < path_list.size(); ++i) {
        const std::string filename = path_list[i].first;
        std::cout << filename << std::endl;
        Network::InT src = load_image(filename, Network::InT::DIM_1);

        timer.start();
        Network::OutT dst = src >> net;
        timer.stop();

        std::vector<int> predict_class;
        std::vector<double> predict_score;
        evaluation<5>(dst, predict_class, predict_score);

        const int label = path_list[i].second;
        tp_counter[TOP1] += (predict_class[0] == label) ? 1 : 0;

        auto iter = std::find(predict_class.begin(), predict_class.end(), label);
        if(iter != predict_class.end()) {
            ++tp_counter[TOP5];
        }

        std::cout
            << "top-5 evaluate class:\n"
                << std::setw(4) << predict_class[0] << "  " << predict_score[0] << ", "
                << std::setw(4) << predict_class[1] << "  " << predict_score[1] << ", "
                << std::setw(4) << predict_class[2] << "  " << predict_score[2] << ", "
                << std::setw(4) << predict_class[3] << "  " << predict_score[3] << ", "
                << std::setw(4) << predict_class[4] << "  " << predict_score[4] << "\n"
            << "run_time[ms]: "
                << std::setprecision(8) << timer.time()
            << std::endl;

        const double conv_run_time =
            net.conv1_1.get_run_time() +
            net.conv1_2.get_run_time() +
            net.conv2_1.get_run_time() +
            net.conv2_2.get_run_time() +
            net.conv3_1.get_run_time() +
            net.conv3_2.get_run_time() +
            net.conv3_3.get_run_time() +
            net.conv4_1.get_run_time() +
            net.conv4_2.get_run_time() +
            net.conv4_3.get_run_time() +
            net.conv5_1.get_run_time() +
            net.conv5_2.get_run_time() +
            net.conv5_3.get_run_time();
        const double fc_run_time =
            net.fc6.get_run_time() +
            net.fc7.get_run_time() +
            net.fc8.get_run_time();

        const std::string save_folder = std::string(argv[ARG_SAVE_FOLDER]).empty()
            ? "./" : std::string(argv[ARG_SAVE_FOLDER]) + "/";
        const bool file_app = true; // true: ’Ç‹L  false: ã‘‚«
        io::save(save_folder + "predict_class.txt", predict_class, 5, file_app);
        io::save(save_folder + "predict_score.txt", predict_score, 5, file_app);
        io::save(save_folder + "net_run_time.txt", timer.time(), file_app);
        io::save(save_folder + "conv_run_time.txt", conv_run_time, file_app);
        io::save(save_folder + "fc_run_time.txt", fc_run_time, file_app);

        io::save(save_folder + "conv1_1_run_time.txt", net.conv1_1.get_run_time(), file_app);
        io::save(save_folder + "conv1_2_run_time.txt", net.conv1_2.get_run_time(), file_app);
        io::save(save_folder + "conv2_1_run_time.txt", net.conv2_1.get_run_time(), file_app);
        io::save(save_folder + "conv2_2_run_time.txt", net.conv2_2.get_run_time(), file_app);
        io::save(save_folder + "conv3_1_run_time.txt", net.conv3_1.get_run_time(), file_app);
        io::save(save_folder + "conv3_2_run_time.txt", net.conv3_2.get_run_time(), file_app);
        io::save(save_folder + "conv3_3_run_time.txt", net.conv3_3.get_run_time(), file_app);
        io::save(save_folder + "conv4_1_run_time.txt", net.conv4_1.get_run_time(), file_app);
        io::save(save_folder + "conv4_2_run_time.txt", net.conv4_2.get_run_time(), file_app);
        io::save(save_folder + "conv4_3_run_time.txt", net.conv4_3.get_run_time(), file_app);
        io::save(save_folder + "conv5_1_run_time.txt", net.conv5_1.get_run_time(), file_app);
        io::save(save_folder + "conv5_2_run_time.txt", net.conv5_2.get_run_time(), file_app);
        io::save(save_folder + "conv5_3_run_time.txt", net.conv5_3.get_run_time(), file_app);
        io::save(save_folder + "fc6_run_time.txt", net.fc6.get_run_time(), file_app);
        io::save(save_folder + "fc7_run_time.txt", net.fc7.get_run_time(), file_app);
        io::save(save_folder + "fc8_run_time.txt", net.fc8.get_run_time(), file_app);
    }
    if(path_list.size() == 0) {
        std::cout << "Load samples is 0." << std::endl;
    }
    else {
        std::cout << "top1 accuracy: " << 1. * tp_counter[TOP1] / path_list.size() << std::endl;
        std::cout << "top5 accuracy: " << 1. * tp_counter[TOP5] / path_list.size() << std::endl;
    }
    return 0;
}