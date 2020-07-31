#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "pointmatcher/PointMatcher.h"

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

int main(int argc, char *argv[]) {
    std::string datadir = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/radar";
    std::string gt = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/gt/radar_odometry.csv";
    std::string config = "/home/keenan/radar_ws/src/yeti/yeti/config/icp.yaml";
    int min_range = 58;
    // int max_points = 10000;
    float zq = 3.0;
    int sigma_gauss = 17;
    // int window_size = 128;
    // float scale = 3.5;
    // int guard_cells = 32;

    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);

    // File for storing the results of ICP on each frame (and the accuracy)
    std::string accfile = "accuracy.csv";
    std::ofstream ofs;
    ofs.open(accfile, std::ios::out);
    ofs << "x,y,yaw,gtx,gty,gtyaw,time1,time2\n";

    float radar_resolution = 0.0432;
    std::vector<int64_t> t1, t2;
    std::vector<float> a1, a2;
    std::vector<bool> v1, v2;
    cv::Mat f1, f2;

    // Create the ICP object
    PM::ICP icp;
    // Use custom ICP parameters
    if (config.empty()) {
           icp.setDefault();
    } else {
        std::ifstream ifs(config.c_str());
        if (!ifs.good()) {
            std::cerr << "Cannot open config file " << config << std::endl; exit(1);
        }
        icp.loadFromYaml(ifs);
    }
    DP::Labels labels;
    labels.push_back(DP::Label("x", 1));
    labels.push_back(DP::Label("y", 1));
    labels.push_back(DP::Label("w", 1));

    DP ref, data;
    double icptime = 0;

    for (uint i = 0; i < radar_files.size() - 1; ++i) {
        if (i == 0) {
            // Read the two radar images
            load_radar(datadir + "/" + radar_files[i], t1, a1, v1, f1);
            load_radar(datadir + "/" + radar_files[i + 1], t2, a2, v2, f2);
            // Extract features
            Eigen::MatrixXf targets1, targets2;
            // cen2019features(f1, max_points, min_range, targets1);
            // cen2019features(f2, max_points, min_range, targets2);
            cen2018features(f1, zq, sigma_gauss, min_range, targets1);
            cen2018features(f2, zq, sigma_gauss, min_range, targets2);
            // Convert targets to cartesian coordinates
            Eigen::MatrixXf cart_targets1, cart_targets2;
            polar_to_cartesian_points(a1, targets1, radar_resolution, cart_targets1);
            polar_to_cartesian_points(a2, targets2, radar_resolution, cart_targets2);
            // Convert to libpointmatcher DataPoint class
            ref = DP(cart_targets1, labels);
            data = DP(cart_targets2, labels);
        } else {
            t1 = t2; a1 = a2; v1 = v2; f1 = f2;
            load_radar(datadir + "/" + radar_files[i + 1], t2, a2, v2, f2);
            Eigen::MatrixXf targets2;
            // cen2019features(f2, max_points, min_range, targets2);
            cen2018features(f2, zq, sigma_gauss, min_range, targets2);
            Eigen::MatrixXf cart_targets2;
            polar_to_cartesian_points(a2, targets2, radar_resolution, cart_targets2);
            ref = data;
            data = DP(cart_targets2, labels);
        }

        // Compute the transformation to express data in ref
        auto start = std::chrono::high_resolution_clock::now();
        PM::TransformationParameters T = icp(data, ref);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e = stop - start;
        icptime += e.count();
        // Retrieve the ground truth to calculate accuracy
        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        int64 time2 = std::stoll(parts[0]);
        std::vector<float> gtvec;
        get_groundtruth_odometry(gt, time1, time2, gtvec);

        float yaw = -1 * asin(T(0, 1));

        ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ",";
        ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[5] << ",";
        ofs << time1 << "," << time2 << "\n";
    }

    icptime /= (radar_files.size() - 1);
    std::cout << "average ICP time: " << icptime << std::endl;

    return 0;
}
