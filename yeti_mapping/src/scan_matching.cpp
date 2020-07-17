#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "pointmatcher/PointMatcher.h"
#include "boost/filesystem.hpp"

inline bool exists(const std::string& name) {
    struct stat buffer;
    return !(stat (name.c_str(), &buffer) == 0);
}

void draw_points(std::string name, cv::Mat x, std::vector<cv::Point> points, cv::Mat &vis) {
    cv::cvtColor(x, vis, cv::COLOR_GRAY2BGR);
    for (cv::Point p : points) {
        cv::circle(vis, p, 1, cv::Scalar(0, 0, 255), -1);
    }
}

void validateArgs(int argc, char *argv[], bool& isCSV )
{
	if (argc != 3)
	{
		std::cerr << "Wrong number of arguments, usage " << argv[0] << " reference.csv reading.csv" << std::endl;
		std::cerr << "Will create 3 vtk files for inspection: ./test_ref.vtk, ./test_data_in.vtk and ./test_data_out.vtk" << std::endl;
		std::cerr << std::endl << "2D Example:" << std::endl;
		std::cerr << "  " << argv[0] << " ../../examples/data/2D_twoBoxes.csv ../../examples/data/2D_oneBox.csv" << std::endl;
		std::cerr << std::endl << "3D Example:" << std::endl;
		std::cerr << "  " << argv[0] << " ../../examples/data/car_cloud400.csv ../../examples/data/car_cloud401.csv" << std::endl;
		exit(1);
	}
}

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

int main(int argc, char *argv[]) {
    bool isCSV = true;
    validateArgs(argc, argv, isCSV);


    std::string datadir = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/radar";
    float cart_resolution = 0.25;
    int cart_pixel_width = 1000;

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
    std::vector<int64_t> t1, t2;
    std::vector<float> a1, a2;
    std::vector<bool> v1, v2;
    cv::Mat f1, f2;

    // Read the two radar images
    load_radar(datadir + "/" + radar_files[100], t1, a1, v1, f1);
    load_radar(datadir + "/" + radar_files[101], t2, a2, v2, f2);

    // Extract features
    Eigen::MatrixXf targets1, targets2;
    int min_range = 58;
    int max_points = 10000;
    cen2019features(f1, max_points, min_range, targets1);
    cen2019features(f2, max_points, min_range, targets2);

    // Visualize targets on top of the cartesian radar images
    cv::Mat cart_img1, cart_img2;
    radar_polar_to_cartesian(a1, f1, radar_resolution, cart_resolution, cart_pixel_width, true, cart_img1);
    radar_polar_to_cartesian(a2, f2, radar_resolution, cart_resolution, cart_pixel_width, true, cart_img2);

    Eigen::MatrixXf cart_targets1, cart_targets2;
    polar_to_cartesian_points(a1, targets1, radar_resolution, cart_targets1);
    polar_to_cartesian_points(a2, targets2, radar_resolution, cart_targets2);
    std::vector<cv::Point> bev_points1, bev_points2;
    convert_to_bev(cart_targets1, cart_resolution, cart_pixel_width, bev_points1);
    convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, bev_points2);

    cv::Mat vis1, vis2;
    draw_points("1", cart_img1, bev_points1, vis1);
    draw_points("2", cart_img2, bev_points2, vis2);

    cv::Mat combined;
    cv::hconcat(vis1, vis2, combined);
    cv::imshow("combo", combined);
    cv::waitKey(0);

    return 0;
}
