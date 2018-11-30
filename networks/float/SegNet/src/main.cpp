#include <cassert>
#include <iomanip>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "network.hpp"
#include "utils/io.hpp"
#include "utils/timer.hpp"
#include "utils/numpy.hpp"


typedef Tensor<int, SegNet::OutT::DIM_2, SegNet::OutT::DIM_3> Map;

void load_list(const std::string & dataset,
               const std::string & basepath,
               std::vector<std::pair<std::string, int> > & path_list)
{
    std::ifstream ifs(dataset);
    std::string line;
    while (getline(ifs, line)) {
        std::vector<std::string> splited_line;
        std::stringstream ss(line);
        std::string buffer;
        while (std::getline(ss, buffer, ' ')) {
            splited_line.push_back(buffer);
        }
        const std::string filename = basepath + (basepath == "" ? "" : "/") + splited_line[0];
        path_list.push_back(
            (splited_line.size() == 2)
                ? std::make_pair(filename, std::stoi(splited_line[1]))
                : std::make_pair(filename, -1)
        );
    }
}

SegNet::InT load_image(const std::string & filename, const int n_channels=1)
{
    const auto color_type = n_channels == 3
        ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE;
    cv::Mat image = cv::imread(filename, color_type);
    if (image.empty()) {
        std::cerr << "CANNOT read files" << std::endl;
        std::exit(-1);
    }
    cv::Mat3d dImage;
    image.convertTo(dImage, (n_channels == 3) ? CV_64FC3 : CV_64FC1);
    cv::resize(dImage, dImage, cv::Size(SegNet::InT::DIM_3, SegNet::InT::DIM_2));

    std::vector<double> data(SegNet::InT::SIZE, 0.0);
    double* p_data = &data[0];
    for (int c = 0; c < dImage.channels(); ++c) {
        for (int y = 0; y < dImage.rows; ++y) {
            for (int x = 0; x < dImage.cols; ++x) {
                //const int data_index = (c * cropped_image.rows + y) * cropped_image.cols + x;
                *p_data++ = static_cast<double>(dImage.at<cv::Vec3d>(y, x)[c]);
            }
        }
    }
    return SegNet::InT(data);
}

Map argmax(const SegNet::OutT& predict)
{
    Map class_map;
    for (int d2 = 0; d2 < SegNet::OutT::DIM_2; ++d2) {
        for (int d3 = 0; d3 < SegNet::OutT::DIM_3; ++d3) {
            double max_value = predict.at(0, d2, d3);
            int& dst = class_map.at(d2, d3);
            dst = 0;
            for (int d1 = 1; d1 < SegNet::OutT::DIM_1; ++d1) {
                const double current_value = predict.at(d1, d2, d3);
                if (max_value < current_value) {
                    max_value = current_value;
                    dst = d1;
                }
            }
        }
    }
    return class_map;
}

void csv_write(const std::string& filename, const int* pred)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::exit(-1);
    }
    for (int d1 = 0; d1 < SegNet::OutT::DIM_2; ++d1) {
        for (int d2 = 0; d2 < SegNet::OutT::DIM_3; ++d2) {
            ofs << *pred++ << ", ";
        }
        ofs << std::endl;
    }
}

cv::Mat3b visualization(const int* map, const std::vector<cv::Vec3b>& palette)
{
    cv::Mat3b canvas(cv::Size(SegNet::OutT::DIM_3, SegNet::OutT::DIM_2), CV_8UC3);
    for (int row = 0; row < canvas.rows; ++row) {
        for (int col = 0; col < canvas.cols; ++col) {
            canvas.at<cv::Vec3b>(row, col) = palette[*map++];
        }
    }
    return canvas;
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

    Timer timer;
    SegNet net(argv[ARG_PARAM_FOLDER]);
    for(size_t i = 0; i < path_list.size(); ++i) {
        const std::string filename = path_list[i].first;
        std::cout << filename << std::endl;
        SegNet::InT src = load_image(filename, SegNet::InT::DIM_1);

        timer.start();
        SegNet::OutT dst = src >> net;
        timer.stop();

        Map class_map = argmax(dst);

        const double conv_time =
            net.conv1_1.get_run_time() + net.conv1_2.get_run_time() +
            net.conv2_1.get_run_time() + net.conv2_2.get_run_time() +
            net.conv3_1.get_run_time() + net.conv3_2.get_run_time() + net.conv3_3.get_run_time() +
            net.conv4_1.get_run_time() + net.conv4_2.get_run_time() + net.conv4_3.get_run_time() +
            net.conv5_1.get_run_time() + net.conv5_2.get_run_time() + net.conv5_3.get_run_time() +
            net.conv5_3_D.get_run_time() + net.conv5_2_D.get_run_time() + net.conv5_1_D.get_run_time() +
            net.conv4_3_D.get_run_time() + net.conv4_2_D.get_run_time() + net.conv4_1_D.get_run_time() +
            net.conv3_3_D.get_run_time() + net.conv3_2_D.get_run_time() + net.conv3_1_D.get_run_time() +
            net.conv2_2_D.get_run_time() + net.conv2_1_D.get_run_time() +
            net.conv1_2_D.get_run_time() + net.conv1_1_D.get_run_time();

        const std::string save_folder = std::string(argv[ARG_SAVE_FOLDER]).empty()
            ? "./" : std::string(argv[ARG_SAVE_FOLDER]) + "/";

        const std::string basename = filename.substr(std::max<signed>(filename.find_last_of('/'), filename.find_last_of('\\')) + 1);
        const std::string dst_name = save_folder + "/map_" + basename + ".npy";
        std::cout << "Output class map: " << dst_name << std::endl;
        aoba::SaveArrayAsNumpy(dst_name, SegNet::OutT::DIM_2, SegNet::OutT::DIM_3, &class_map.data[0]);
        //csv_write(dst_name, &class_map.data[0]);
        //cv::imwrite(dst_name, visualization(&class_map.data[0], palette));
        std::cout << "test time[ms]: " << std::setprecision(4) << timer.time() << std::endl;

        const bool file_app = true; // true: ’Ç‹L  false: ã‘‚«
        io::save(save_folder + "test_time.txt", timer.time(), file_app);
        io::save(save_folder + "conv_time.txt", conv_time, file_app);

        io::save(save_folder + "conv1_1_time.txt", net.conv1_1.get_run_time(), file_app);
        io::save(save_folder + "conv1_2_time.txt", net.conv1_2.get_run_time(), file_app);
        io::save(save_folder + "conv2_1_time.txt", net.conv2_1.get_run_time(), file_app);
        io::save(save_folder + "conv2_2_time.txt", net.conv2_2.get_run_time(), file_app);
        io::save(save_folder + "conv3_1_time.txt", net.conv3_1.get_run_time(), file_app);
        io::save(save_folder + "conv3_2_time.txt", net.conv3_2.get_run_time(), file_app);
        io::save(save_folder + "conv3_3_time.txt", net.conv3_3.get_run_time(), file_app);
        io::save(save_folder + "conv4_1_time.txt", net.conv4_1.get_run_time(), file_app);
        io::save(save_folder + "conv4_2_time.txt", net.conv4_2.get_run_time(), file_app);
        io::save(save_folder + "conv4_3_time.txt", net.conv4_3.get_run_time(), file_app);
        io::save(save_folder + "conv5_1_time.txt", net.conv5_1.get_run_time(), file_app);
        io::save(save_folder + "conv5_2_time.txt", net.conv5_2.get_run_time(), file_app);
        io::save(save_folder + "conv5_3_time.txt", net.conv5_3.get_run_time(), file_app);

        io::save(save_folder + "conv5_3_D_time.txt", net.conv5_3_D.get_run_time(), file_app);
        io::save(save_folder + "conv5_2_D_time.txt", net.conv5_2_D.get_run_time(), file_app);
        io::save(save_folder + "conv5_1_D_time.txt", net.conv5_1_D.get_run_time(), file_app);
        io::save(save_folder + "conv4_3_D_time.txt", net.conv4_3_D.get_run_time(), file_app);
        io::save(save_folder + "conv4_2_D_time.txt", net.conv4_2_D.get_run_time(), file_app);
        io::save(save_folder + "conv4_1_D_time.txt", net.conv4_1_D.get_run_time(), file_app);
        io::save(save_folder + "conv3_3_D_time.txt", net.conv3_3_D.get_run_time(), file_app);
        io::save(save_folder + "conv3_2_D_time.txt", net.conv3_2_D.get_run_time(), file_app);
        io::save(save_folder + "conv3_1_D_time.txt", net.conv3_1_D.get_run_time(), file_app);
        io::save(save_folder + "conv2_2_D_time.txt", net.conv2_2_D.get_run_time(), file_app);
        io::save(save_folder + "conv2_1_D_time.txt", net.conv2_1_D.get_run_time(), file_app);
        io::save(save_folder + "conv1_2_D_time.txt", net.conv1_2_D.get_run_time(), file_app);
        io::save(save_folder + "conv1_1_D_time.txt", net.conv1_1_D.get_run_time(), file_app);
    }
    if(path_list.size() == 0) {
        std::cout << "Load samples is 0." << std::endl;
    }
    return 0;
}