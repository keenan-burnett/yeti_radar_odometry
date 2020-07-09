#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"
#include "features.hpp"

inline bool exists(const std::string& name) {
    struct stat buffer;
    return !(stat (name.c_str(), &buffer) == 0);
}

int main() {
    std::string datadir = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/radar";
    float cart_resolution = 0.25;
    int cart_pixel_width = 1000;
    bool interpolate_crossover = true;

    // Get file names of the radar images
    DIR *dirp = opendir(datadir.c_str());
    struct dirent *dp;
    std::vector<std::string> radar_files;
    while ((dp = readdir(dirp)) != NULL) {
        if (exists(dp->d_name))
            radar_files.push_back(dp->d_name);
    }
    // Sort files in ascending order of time stamp
    std::sort(radar_files.begin(), radar_files.end(), less_than_img());

    float radar_resolution = 0.0432;
    std::vector<int64_t> timestamps;
    std::vector<float> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    load_radar(datadir + "/" + radar_files[100], timestamps, azimuths, valid, fft_data);

    cv::Mat cart_img;
    radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
        interpolate_crossover, cart_img);

    std::vector<cv::Point2f> targets;

    // int window_size = 128;
    // float scale = 3.5;
    // int guard_cells = 32;
    // cfar1d(fft_data, window_size, scale, guard_cells, targets);

    float zq = 2.5;
    int w_median = 200;
    int sigma_gauss = 17;
    int min_range = 58;
    cen2018features(fft_data, zq, w_median, sigma_gauss, min_range, targets);

    std::cout << "targets: " << targets.size() << std::endl;

    std::vector<cv::Point2f> cart_targets;
    polar_to_cartesian_points(azimuths, targets, radar_resolution, cart_targets);
    std::vector<cv::Point> bev_points;
    convert_to_bev(cart_targets, cart_resolution, cart_pixel_width, bev_points);

    cv::Mat vis;
    cv::cvtColor(cart_img, vis, cv::COLOR_GRAY2BGR);
    for (cv::Point p : bev_points) {
        cv::circle(vis, p, 1, cv::Scalar(0, 0, 255), -1);
    }
    cv::imshow("cart", vis);
    cv::waitKey(0);

    return 0;
}
