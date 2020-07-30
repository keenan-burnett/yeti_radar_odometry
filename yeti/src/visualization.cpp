#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"
#include "features.hpp"

int main() {
    std::string datadir = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/radar";
    float cart_resolution = 0.25;
    int cart_pixel_width = 1000;
    bool interpolate_crossover = true;

    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);

    float radar_resolution = 0.0432;
    std::vector<int64_t> timestamps;
    std::vector<float> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    load_radar(datadir + "/" + radar_files[100], timestamps, azimuths, valid, fft_data);

    cv::Mat cart_img;
    radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
        interpolate_crossover, cart_img);

    Eigen::MatrixXf targets;
    int min_range = 58;

    // int window_size = 128;
    // float scale = 3.5;
    // int guard_cells = 32;
    // cfar1d(fft_data, window_size, scale, guard_cells, min_range, targets);

    // float zq = 2.5;
    // int sigma_gauss = 17;
    // cen2018features(fft_data, zq, sigma_gauss, min_range, targets);

    int max_points = 20000;
    cen2019features(fft_data, max_points, min_range, targets);

    std::cout << "targets: " << targets.cols() << std::endl;

    Eigen::MatrixXf cart_targets;
    polar_to_cartesian_points(azimuths, targets, radar_resolution, cart_targets);
    cv::Mat vis;
    draw_points(cart_img, cart_targets, cart_resolution, cart_pixel_width, vis);
    cv::imshow("cart", vis);
    cv::waitKey(0);

    return 0;
}
