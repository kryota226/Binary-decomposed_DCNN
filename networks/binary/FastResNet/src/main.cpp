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
    std::vector<std::pair<std::string, int> > & data_list
) {
    data_list.clear();
    std::ifstream ifs(dataset);
    while(ifs.peek() != std::ios::traits_type::eof()) {
        std::string filename;
        int label;
        ifs >> filename >> label;
        filename = basepath + (basepath == "" ? "" : "/") + filename;
        data_list.push_back(std::make_pair(filename, label));
    }
    if (data_list.empty()) {
        std::cerr << "data_list is empty." << std::endl;
        std::exit(-1);
    }
}

void preprocess(
	const cv::Mat3d & dImage, const cv::Mat3d & mean_image, cv::Mat3d & cropped_image)
{
	const int target_size = 256;
	const int xh = dImage.rows;
	const int xw = dImage.cols;
	const int out_h = xw < xh ? target_size * xh / xw : target_size;
	const int out_w = xw > xh ? target_size * xw / xh : target_size;
	cv::Mat3d target_image;
	cv::resize(dImage, target_image, cv::Size(out_w, out_h));
	const int start_h = (out_h - target_size) / 2;
	const int start_w = (out_w - target_size) / 2;
	cv::Rect target_roi(start_w, start_h, target_size, target_size);
	target_image = target_image(target_roi);

	const int start = (target_size - Network::InT::DIM_2) / 2;
	cv::Rect crop_size(start, start, Network::InT::DIM_3, Network::InT::DIM_2);
	cropped_image = target_image(crop_size) - mean_image(crop_size);
}

Network::InT load_image(
	const std::string & filename, const int n_channels, const cv::Mat3d & mean_image
) {
	const auto load_color = n_channels == 3
		? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
	cv::Mat3b image = cv::imread(filename, load_color);
	if (image.empty()) {
		std::cerr << "CANNOT read files" << std::endl;
		std::exit(-1);
	}
	cv::Mat3d dImage;
	image.convertTo(dImage, CV_64FC3);

	cv::Mat3d cropped_image;
	preprocess(dImage, mean_image, cropped_image);

    std::vector<double> data(Network::InT::SIZE, 0.0);
    double* p_data = &data[0];
    for (int c = 0; c < cropped_image.channels(); ++c) {
        for (int y = 0; y < cropped_image.rows; ++y) {
            for (int x = 0; x < cropped_image.cols; ++x) {
                //const int data_index = (c * cropped_image.rows + y) * cropped_image.cols + x;
                *p_data++ = static_cast<double>(cropped_image.at<cv::Vec3d>(y, x)[c]);
            }
        }
    }
    return Network::InT(data);
}


template <int n_orders>
void evaluation(
    const Network::OutT & prob,
    std::vector<int> & evaluated,
    std::vector<double> & score
) {
    assert(n_orders <= Network::OutT::SIZE);
    std::vector<std::pair<double, int>> prob_index;
    for(int i = 0; i < Network::OutT::SIZE; ++i) {
		prob_index.push_back(std::make_pair(prob.at(i), i));
    }
    std::partial_sort(
        prob_index.begin(),
        prob_index.begin() + n_orders,
        prob_index.end(),
        std::greater<std::pair<double, int> >()
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
		ARG_MEAN_FILE,
		ARG_PARAM_FOLDER,
        ARG_SAVE_FOLDER,
        NUM_ARGS,
    };
    assert(NUM_ARGS <= argc);

    std::vector<std::pair<std::string, int> > data_list;
    load_list(argv[ARG_DATASET], argv[ARG_BASE_PATH], data_list);

	std::vector<double> mean_data = io::load<double>(argv[ARG_MEAN_FILE]);
	cv::Mat3d mean_image = cv::Mat(256, 256, CV_64FC3, &mean_data[0]);

	const std::string save_folder = std::string(argv[ARG_SAVE_FOLDER]).empty()
		? "./" : std::string(argv[ARG_SAVE_FOLDER]) + "/";

    enum { TOP1, TOP5, N_EVALS };
    std::vector<int> tp_counter(N_EVALS, 0);
    Timer timer;
    Network net(argv[ARG_PARAM_FOLDER]);

    // warming up
#if 0
    {
		const std::string filename = path_list[0].first;
		Network::InT src = load_image(filename, Network::InT::DIM_1, mean_image);
		for (size_t i = 0; i < 10; ++i) {
			src >> net;
		}
    }
#endif

    std::cout << "# of test samples: " << data_list.size() << std::endl;
    for (size_t i = 0; i < data_list.size(); ++i) {
        const std::string filename = data_list[i].first;
        std::cout << filename << std::endl;
		Network::InT src = load_image(filename, Network::InT::DIM_1, mean_image);

        timer.start();
        Network::OutT dst = src >> net;
        timer.stop();

        std::vector<int> predict_class;
        std::vector<double> predict_score;
        evaluation<5>(dst, predict_class, predict_score);

        const int label = data_list[i].second;
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
                << std::setprecision(4) << timer.time()
            << std::endl;

        const double conv_time =
            net.conv1.get_run_time() +
			net.res2a_branch1.get_run_time() +
			net.res2a_branch2a.get_run_time() +
			net.res2a_branch2b.get_run_time() +
			net.res2a_branch2c.get_run_time() +
			net.res2b_branch2a.get_run_time() +
			net.res2b_branch2b.get_run_time() +
			net.res2b_branch2c.get_run_time() +
			net.res2c_branch2a.get_run_time() +
			net.res2c_branch2b.get_run_time() +
			net.res2c_branch2c.get_run_time() +

			net.res3a_branch1.get_run_time() +
			net.res3a_branch2a.get_run_time() +
			net.res3a_branch2b.get_run_time() +
			net.res3a_branch2c.get_run_time() +
			net.res3b1_branch2a.get_run_time() +
			net.res3b1_branch2b.get_run_time() +
			net.res3b1_branch2c.get_run_time() +
			net.res3b2_branch2a.get_run_time() +
			net.res3b2_branch2b.get_run_time() +
			net.res3b2_branch2c.get_run_time() +
			net.res3b3_branch2a.get_run_time() +
			net.res3b3_branch2b.get_run_time() +
			net.res3b3_branch2c.get_run_time() +
			net.res3b4_branch2a.get_run_time() +
			net.res3b4_branch2b.get_run_time() +
			net.res3b4_branch2c.get_run_time() +
			net.res3b5_branch2a.get_run_time() +
			net.res3b5_branch2b.get_run_time() +
			net.res3b5_branch2c.get_run_time() +
			net.res3b6_branch2a.get_run_time() +
			net.res3b6_branch2b.get_run_time() +
			net.res3b6_branch2c.get_run_time() +
			net.res3b7_branch2a.get_run_time() +
			net.res3b7_branch2b.get_run_time() +
			net.res3b7_branch2c.get_run_time() +

			net.res4a_branch1.get_run_time() +
			net.res4a_branch2a.get_run_time() +
			net.res4a_branch2b.get_run_time() +
			net.res4a_branch2c.get_run_time() +
			net.res4b1_branch2a.get_run_time() +
			net.res4b1_branch2b.get_run_time() +
			net.res4b1_branch2c.get_run_time() +
			net.res4b2_branch2a.get_run_time() +
			net.res4b2_branch2b.get_run_time() +
			net.res4b2_branch2c.get_run_time() +
			net.res4b3_branch2a.get_run_time() +
			net.res4b3_branch2b.get_run_time() +
			net.res4b3_branch2c.get_run_time() +
			net.res4b4_branch2a.get_run_time() +
			net.res4b4_branch2b.get_run_time() +
			net.res4b4_branch2c.get_run_time() +
			net.res4b5_branch2a.get_run_time() +
			net.res4b5_branch2b.get_run_time() +
			net.res4b5_branch2c.get_run_time() +
			net.res4b6_branch2a.get_run_time() +
			net.res4b6_branch2b.get_run_time() +
			net.res4b6_branch2c.get_run_time() +
			net.res4b7_branch2a.get_run_time() +
			net.res4b7_branch2b.get_run_time() +
			net.res4b7_branch2c.get_run_time() +
			net.res4b8_branch2a.get_run_time() +
			net.res4b8_branch2b.get_run_time() +
			net.res4b8_branch2c.get_run_time() +
			net.res4b9_branch2a.get_run_time() +
			net.res4b9_branch2b.get_run_time() +
			net.res4b9_branch2c.get_run_time() +
			net.res4b10_branch2a.get_run_time() +
			net.res4b10_branch2b.get_run_time() +
			net.res4b10_branch2c.get_run_time() +
			net.res4b11_branch2a.get_run_time() +
			net.res4b11_branch2b.get_run_time() +
			net.res4b11_branch2c.get_run_time() +
			net.res4b12_branch2a.get_run_time() +
			net.res4b12_branch2b.get_run_time() +
			net.res4b12_branch2c.get_run_time() +
			net.res4b13_branch2a.get_run_time() +
			net.res4b13_branch2b.get_run_time() +
			net.res4b13_branch2c.get_run_time() +
			net.res4b14_branch2a.get_run_time() +
			net.res4b14_branch2b.get_run_time() +
			net.res4b14_branch2c.get_run_time() +
			net.res4b15_branch2a.get_run_time() +
			net.res4b15_branch2b.get_run_time() +
			net.res4b15_branch2c.get_run_time() +
			net.res4b16_branch2a.get_run_time() +
			net.res4b16_branch2b.get_run_time() +
			net.res4b16_branch2c.get_run_time() +
			net.res4b17_branch2a.get_run_time() +
			net.res4b17_branch2b.get_run_time() +
			net.res4b17_branch2c.get_run_time() +
			net.res4b18_branch2a.get_run_time() +
			net.res4b18_branch2b.get_run_time() +
			net.res4b18_branch2c.get_run_time() +
			net.res4b19_branch2a.get_run_time() +
			net.res4b19_branch2b.get_run_time() +
			net.res4b19_branch2c.get_run_time() +
			net.res4b20_branch2a.get_run_time() +
			net.res4b20_branch2b.get_run_time() +
			net.res4b20_branch2c.get_run_time() +
			net.res4b21_branch2a.get_run_time() +
			net.res4b21_branch2b.get_run_time() +
			net.res4b21_branch2c.get_run_time() +
			net.res4b22_branch2a.get_run_time() +
			net.res4b22_branch2b.get_run_time() +
			net.res4b22_branch2c.get_run_time() +
			net.res4b23_branch2a.get_run_time() +
			net.res4b23_branch2b.get_run_time() +
			net.res4b23_branch2c.get_run_time() +
			net.res4b24_branch2a.get_run_time() +
			net.res4b24_branch2b.get_run_time() +
			net.res4b24_branch2c.get_run_time() +
			net.res4b25_branch2a.get_run_time() +
			net.res4b25_branch2b.get_run_time() +
			net.res4b25_branch2c.get_run_time() +
			net.res4b26_branch2a.get_run_time() +
			net.res4b26_branch2b.get_run_time() +
			net.res4b26_branch2c.get_run_time() +
			net.res4b27_branch2a.get_run_time() +
			net.res4b27_branch2b.get_run_time() +
			net.res4b27_branch2c.get_run_time() +
			net.res4b28_branch2a.get_run_time() +
			net.res4b28_branch2b.get_run_time() +
			net.res4b28_branch2c.get_run_time() +
			net.res4b29_branch2a.get_run_time() +
			net.res4b29_branch2b.get_run_time() +
			net.res4b29_branch2c.get_run_time() +
			net.res4b30_branch2a.get_run_time() +
			net.res4b30_branch2b.get_run_time() +
			net.res4b30_branch2c.get_run_time() +
			net.res4b31_branch2a.get_run_time() +
			net.res4b31_branch2b.get_run_time() +
			net.res4b31_branch2c.get_run_time() +
			net.res4b32_branch2a.get_run_time() +
			net.res4b32_branch2b.get_run_time() +
			net.res4b32_branch2c.get_run_time() +
			net.res4b33_branch2a.get_run_time() +
			net.res4b33_branch2b.get_run_time() +
			net.res4b33_branch2c.get_run_time() +
			net.res4b34_branch2a.get_run_time() +
			net.res4b34_branch2b.get_run_time() +
			net.res4b34_branch2c.get_run_time() +
			net.res4b35_branch2a.get_run_time() +
			net.res4b35_branch2b.get_run_time() +
			net.res4b35_branch2c.get_run_time() +

			net.res5a_branch1.get_run_time() +
			net.res5a_branch2a.get_run_time() +
			net.res5a_branch2b.get_run_time() +
			net.res5a_branch2c.get_run_time() +
			net.res5b_branch2a.get_run_time() +
			net.res5b_branch2b.get_run_time() +
			net.res5b_branch2c.get_run_time() +
			net.res5c_branch2a.get_run_time() +
			net.res5c_branch2b.get_run_time() +
			net.res5c_branch2c.get_run_time();

        const double fc_time =
            net.fc1000.get_run_time();

        const bool file_app = true; // true: ’Ç‹L  false: ã‘‚«
        io::save(save_folder + "predict_class.txt", predict_class, 5, file_app);
        io::save(save_folder + "predict_score.txt", predict_score, 5, file_app);
        io::save(save_folder + "test_time.txt", timer.time(), file_app);
        io::save(save_folder + "conv_time.txt", conv_time, file_app);
        io::save(save_folder + "fc_time.txt", fc_time, file_app);

        io::save(save_folder + "conv1_run_time.txt", net.conv1.get_run_time(), file_app);
		io::save(save_folder + "res2a_branch1_time.txt", net.res2a_branch1.get_run_time(), file_app);
		io::save(save_folder + "res2a_branch2a_time.txt", net.res2a_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res2a_branch2b_time.txt", net.res2a_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res2a_branch2c_time.txt", net.res2a_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res2b_branch2a_time.txt", net.res2a_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res2b_branch2b_time.txt", net.res2a_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res2b_branch2c_time.txt", net.res2a_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res2c_branch2a_time.txt", net.res2a_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res2c_branch2b_time.txt", net.res2a_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res2c_branch2c_time.txt", net.res2a_branch2c.get_run_time(), file_app);

		io::save(save_folder + "res3a_branch1_time.txt", net.res3a_branch1.get_run_time(), file_app);
		io::save(save_folder + "res3a_branch2a_time.txt", net.res3a_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res3a_branch2b_time.txt", net.res3a_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res3a_branch2c_time.txt", net.res3a_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res3b1_branch2a_time.txt", net.res3b1_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res3b1_branch2b_time.txt", net.res3b1_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res3b1_branch2c_time.txt", net.res3b1_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res3b2_branch2a_time.txt", net.res3b2_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res3b2_branch2b_time.txt", net.res3b2_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res3b2_branch2c_time.txt", net.res3b2_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res3b3_branch2a_time.txt", net.res3b3_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res3b3_branch2b_time.txt", net.res3b3_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res3b3_branch2c_time.txt", net.res3b3_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res3b4_branch2a_time.txt", net.res3b4_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res3b4_branch2b_time.txt", net.res3b4_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res3b4_branch2c_time.txt", net.res3b4_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res3b5_branch2a_time.txt", net.res3b5_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res3b5_branch2b_time.txt", net.res3b5_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res3b5_branch2c_time.txt", net.res3b5_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res3b6_branch2a_time.txt", net.res3b6_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res3b6_branch2b_time.txt", net.res3b6_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res3b6_branch2c_time.txt", net.res3b6_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res3b7_branch2a_time.txt", net.res3b7_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res3b7_branch2b_time.txt", net.res3b7_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res3b7_branch2c_time.txt", net.res3b7_branch2c.get_run_time(), file_app);

		io::save(save_folder + "res4a_branch1_time.txt", net.res4a_branch1.get_run_time(), file_app);
		io::save(save_folder + "res4a_branch2a_time.txt", net.res4a_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4a_branch2b_time.txt", net.res4a_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4a_branch2c_time.txt", net.res4a_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b1_branch2a_time.txt", net.res4b1_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b1_branch2b_time.txt", net.res4b1_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b1_branch2c_time.txt", net.res4b1_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b2_branch2a_time.txt", net.res4b2_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b2_branch2b_time.txt", net.res4b2_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b2_branch2c_time.txt", net.res4b2_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b3_branch2a_time.txt", net.res4b3_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b3_branch2b_time.txt", net.res4b3_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b3_branch2c_time.txt", net.res4b3_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b4_branch2a_time.txt", net.res4b4_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b4_branch2b_time.txt", net.res4b4_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b4_branch2c_time.txt", net.res4b4_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b5_branch2a_time.txt", net.res4b5_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b5_branch2b_time.txt", net.res4b5_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b5_branch2c_time.txt", net.res4b5_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b6_branch2a_time.txt", net.res4b6_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b6_branch2b_time.txt", net.res4b6_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b6_branch2c_time.txt", net.res4b6_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b7_branch2a_time.txt", net.res4b7_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b7_branch2b_time.txt", net.res4b7_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b7_branch2c_time.txt", net.res4b7_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b8_branch2a_time.txt", net.res4b8_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b8_branch2b_time.txt", net.res4b8_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b8_branch2c_time.txt", net.res4b8_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b9_branch2a_time.txt", net.res4b9_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b9_branch2b_time.txt", net.res4b9_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b9_branch2c_time.txt", net.res4b9_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b10_branch2a_time.txt", net.res4b10_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b10_branch2b_time.txt", net.res4b10_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b10_branch2c_time.txt", net.res4b10_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b11_branch2a_time.txt", net.res4b11_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b11_branch2b_time.txt", net.res4b11_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b11_branch2c_time.txt", net.res4b11_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b12_branch2a_time.txt", net.res4b12_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b12_branch2b_time.txt", net.res4b12_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b12_branch2c_time.txt", net.res4b12_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b13_branch2a_time.txt", net.res4b13_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b13_branch2b_time.txt", net.res4b13_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b13_branch2c_time.txt", net.res4b13_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b14_branch2a_time.txt", net.res4b14_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b14_branch2b_time.txt", net.res4b14_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b14_branch2c_time.txt", net.res4b14_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b15_branch2a_time.txt", net.res4b15_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b15_branch2b_time.txt", net.res4b15_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b15_branch2c_time.txt", net.res4b15_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b16_branch2a_time.txt", net.res4b16_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b16_branch2b_time.txt", net.res4b16_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b16_branch2c_time.txt", net.res4b16_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b17_branch2a_time.txt", net.res4b17_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b17_branch2b_time.txt", net.res4b17_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b17_branch2c_time.txt", net.res4b17_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b18_branch2a_time.txt", net.res4b18_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b18_branch2b_time.txt", net.res4b18_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b18_branch2c_time.txt", net.res4b18_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b19_branch2a_time.txt", net.res4b19_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b19_branch2b_time.txt", net.res4b19_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b19_branch2c_time.txt", net.res4b19_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b20_branch2a_time.txt", net.res4b20_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b20_branch2b_time.txt", net.res4b20_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b20_branch2c_time.txt", net.res4b20_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b21_branch2a_time.txt", net.res4b21_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b21_branch2b_time.txt", net.res4b21_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b21_branch2c_time.txt", net.res4b21_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b22_branch2a_time.txt", net.res4b22_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b22_branch2b_time.txt", net.res4b22_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b22_branch2c_time.txt", net.res4b22_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b23_branch2a_time.txt", net.res4b23_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b23_branch2b_time.txt", net.res4b23_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b23_branch2c_time.txt", net.res4b23_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b24_branch2a_time.txt", net.res4b24_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b24_branch2b_time.txt", net.res4b24_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b24_branch2c_time.txt", net.res4b24_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b25_branch2a_time.txt", net.res4b25_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b25_branch2b_time.txt", net.res4b25_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b25_branch2c_time.txt", net.res4b25_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b26_branch2a_time.txt", net.res4b26_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b26_branch2b_time.txt", net.res4b26_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b26_branch2c_time.txt", net.res4b26_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b27_branch2a_time.txt", net.res4b27_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b27_branch2b_time.txt", net.res4b27_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b27_branch2c_time.txt", net.res4b27_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b28_branch2a_time.txt", net.res4b28_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b28_branch2b_time.txt", net.res4b28_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b28_branch2c_time.txt", net.res4b28_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b29_branch2a_time.txt", net.res4b29_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b29_branch2b_time.txt", net.res4b29_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b29_branch2c_time.txt", net.res4b29_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b30_branch2a_time.txt", net.res4b30_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b30_branch2b_time.txt", net.res4b30_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b30_branch2c_time.txt", net.res4b30_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b31_branch2a_time.txt", net.res4b31_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b31_branch2b_time.txt", net.res4b31_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b31_branch2c_time.txt", net.res4b31_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b32_branch2a_time.txt", net.res4b32_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b32_branch2b_time.txt", net.res4b32_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b32_branch2c_time.txt", net.res4b32_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b33_branch2a_time.txt", net.res4b33_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b33_branch2b_time.txt", net.res4b33_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b33_branch2c_time.txt", net.res4b33_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b34_branch2a_time.txt", net.res4b34_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b34_branch2b_time.txt", net.res4b34_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b34_branch2c_time.txt", net.res4b34_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res4b35_branch2a_time.txt", net.res4b35_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res4b35_branch2b_time.txt", net.res4b35_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res4b35_branch2c_time.txt", net.res4b35_branch2c.get_run_time(), file_app);

		io::save(save_folder + "res5a_branch1_time.txt", net.res5a_branch1.get_run_time(), file_app);
		io::save(save_folder + "res5a_branch2a_time.txt", net.res5a_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res5a_branch2b_time.txt", net.res5a_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res5a_branch2c_time.txt", net.res5a_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res5b_branch2a_time.txt", net.res5b_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res5b_branch2b_time.txt", net.res5b_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res5b_branch2c_time.txt", net.res5b_branch2c.get_run_time(), file_app);
		io::save(save_folder + "res5c_branch2a_time.txt", net.res5c_branch2a.get_run_time(), file_app);
		io::save(save_folder + "res5c_branch2b_time.txt", net.res5c_branch2b.get_run_time(), file_app);
		io::save(save_folder + "res5c_branch2c_time.txt", net.res5c_branch2c.get_run_time(), file_app);

		io::save(save_folder + "fc1000_run_time.txt", net.fc1000.get_run_time(), file_app);
    }
    std::cout << "top1 accuracy: " << 1. * tp_counter[TOP1] / data_list.size() << std::endl;
    std::cout << "top5 accuracy: " << 1. * tp_counter[TOP5] / data_list.size() << std::endl;

    return 0;
}