#include <iostream>
#include <string>
#include <fstream>
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
    std::string config = "/home/keenan/radar_ws/src/yeti/yeti/config/icp.yaml";
    float cart_resolution = 0.25;
    int cart_pixel_width = 1000;

    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);

    float radar_resolution = 0.0432;
    std::vector<int64_t> t1, t2;
    std::vector<float> a1, a2;
    std::vector<bool> v1, v2;
    cv::Mat f1, f2;

    // Read the two radar images
    load_radar(datadir + "/" + radar_files[0], t1, a1, v1, f1);
    load_radar(datadir + "/" + radar_files[1], t2, a2, v2, f2);

    std::cout << "R1: " << radar_files[0] << std::endl;
    std::cout << "R2: " << radar_files[1] << std::endl;

    // Extract features
    Eigen::MatrixXf targets1, targets2;
    int min_range = 58;
    int max_points = 10000;
    cen2019features(f1, max_points, min_range, targets1);
    cen2019features(f2, max_points, min_range, targets2);

    // Convert targets to cartesian coordinates
    Eigen::MatrixXf cart_targets1, cart_targets2;
    polar_to_cartesian_points(a1, targets1, radar_resolution, cart_targets1);
    polar_to_cartesian_points(a2, targets2, radar_resolution, cart_targets2);

    // Visualize targets on top of the cartesian radar images
    cv::Mat cart_img1, cart_img2;
    radar_polar_to_cartesian(a1, f1, radar_resolution, cart_resolution, cart_pixel_width, true, cart_img1);
    radar_polar_to_cartesian(a2, f2, radar_resolution, cart_resolution, cart_pixel_width, true, cart_img2);;
    cv::Mat vis1, vis2, combined;
    draw_points(cart_img1, cart_targets1, cart_resolution, cart_pixel_width, vis1);
    draw_points(cart_img2, cart_targets2, cart_resolution, cart_pixel_width, vis2);
    cv::hconcat(vis1, vis2, combined);
    cv::imshow("combo", combined);
    cv::waitKey(0);

    // Convert to libpointmatcher DataPoint class
    DP::Labels labels;
    labels.push_back(DP::Label("x", 1));
    labels.push_back(DP::Label("y", 1));
    labels.push_back(DP::Label("w", 1));
    DP ref(cart_targets1, labels);
    DP data(cart_targets2, labels);

    // Create the default ICP algorithm
    PM::ICP icp;
    // See the implementation of setDefault() to create a custom ICP algorithm
    if (config.empty()) {
           icp.setDefault();
    } else {
        std::ifstream ifs(config.c_str());
        if (!ifs.good()) {
            std::cerr << "Cannot open config file " << config << std::endl; exit(1);
        }
        icp.loadFromYaml(ifs);
    }

    // Compute the transformation to express data in ref
    PM::TransformationParameters T = icp(data, ref);

    // Transform data to express it in ref
    DP data_out(data);
    icp.transformations.apply(data_out, T);

    // Safe files to see the results
    ref.save("test_ref.vtk");
    data.save("test_data_in.vtk");
    data_out.save("test_data_out.vtk");
    std::cout << "Final transformation:" << std::endl << T << std::endl;

    // std::cout << "Ground truth: "

    return 0;
}
