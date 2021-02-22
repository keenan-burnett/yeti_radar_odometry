#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"

int main(int argc, const char *argv[]) {
    std::string root;
    validateArgs(argc, argv, root);
    std::cout << root << std::endl;
    float cart_resolution = 0.2240;
    int cart_pixel_width = 640;
    bool interpolate_crossover = true;
    float radar_resolution = 0.0560;
    int navtech_version = CIR204;
    std::vector<std::string> radar_files;
    std::string radar_folder = root + "radar/";
    std::string output_folder = radar_folder + "cart/";
    omp_set_num_threads(8);
    get_file_names(radar_folder, radar_files, "png");
    for (uint i = 0; i < radar_files.size(); ++i) {
        std::cout << i << " / " << radar_files.size() - 1 << std::endl;
        std::vector<int64_t> timestamps;
        std::vector<double> azimuths;
        std::vector<bool> valid;
        cv::Mat fft_data;
        load_radar(radar_folder + radar_files[i], timestamps, azimuths, valid, fft_data, navtech_version);
        cv::Mat cart_img;
        radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
            interpolate_crossover, cart_img, navtech_version);
        cv::imwrite(output_folder + radar_files[i], cart_img);
    }
    return 0;
}
