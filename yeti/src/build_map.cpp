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
using namespace PointMatcherSupport;  // NOLINT

int main(int argc, char *argv[]) {
    std::string datadir = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/radar";
    std::string gt = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/gt/radar_odometry.csv";
    std::string config = "/home/keenan/radar_ws/src/yeti/yeti/config/icp.yaml";
    std::string map_name = "map.vtk";
    int min_range = 58;
    // int max_points = 10000;
    float zq = 4.0;
    int sigma_gauss = 17;

    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);

    // File for storing the results of ICP on each frame (and the accuracy)
    std::string accfile = "pose.csv";
    std::ofstream ofs;
    ofs.open(accfile, std::ios::out);
    ofs << "T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,time1,time2\n";

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

    std::shared_ptr<PM::DataPointsFilter> quadTreeSubsample =
        PM::get().DataPointsFilterRegistrar.create("OctreeGridDataPointsFilter", {{"maxSizeByNode", "0.25"}, {"samplingMethod", "1"}});  // NOLINT

    DP map, newCloud, ref, transformed;
    double icptime = 0;
    // Rigid transformation
    std::shared_ptr<PM::Transformation> rigidTrans;
    rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");
    PM::TransformationParameters T = Eigen::Matrix3f::Identity();
    PM::TransformationParameters T_map_new = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f prior = Eigen::Matrix3f::Identity();

    // for (uint i = 0; i < radar_files.size() - 1; ++i) {
    for (uint i = 0; i < 50; ++i) {
        std::cout << i << std::endl;
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
            map = DP(cart_targets1, labels);
            // ref = DP(cart_targets1, labels);
            newCloud = DP(cart_targets2, labels);
        } else {
            t1 = t2; a1 = a2; v1 = v2; f1 = f2;
            load_radar(datadir + "/" + radar_files[i + 1], t2, a2, v2, f2);
            Eigen::MatrixXf targets2;
            // cen2019features(f2, max_points, min_range, targets2);
            cen2018features(f2, zq, sigma_gauss, min_range, targets2);
            Eigen::MatrixXf cart_targets2;
            polar_to_cartesian_points(a2, targets2, radar_resolution, cart_targets2);
            // ref = newCloud;
            newCloud = DP(cart_targets2, labels);
        }

        // Compute the transformation to express data in ref
        auto start = std::chrono::high_resolution_clock::now();
        PM::TransformationParameters T = icp(newCloud, map, prior);
        T = rigidTrans->correctParameters(T);    // Useful if the same matrix is composed in a loop
        prior = T;
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e = stop - start;
        icptime += e.count();

        // Move the new point cloud in the map reference
        // T_map_new = T * T_map_new;
        // T_map_new = rigidTrans->correctParameters(T_map_new);
        transformed = rigidTrans->compute(newCloud, T);

        std::cout << transformed.getNbPoints() << std::endl;

        icp.matcher->init(map);
        PM::Matches matches = icp.matcher->findClosests(transformed);
        PM::OutlierWeights outlierWeights = icp.outlierFilters.compute(transformed, map, matches);
        // float error = icp.errorMinimizer->getResidualError(transformed, map, outlierWeights, matches);

        int y = 0;
        for (int x = 0; x < outlierWeights.cols(); ++x) {
            if (outlierWeights(0, x)) {
                transformed.setColFrom(y, transformed, x);
                y++;
            }
            x++;
        }
        transformed.conservativeResize(y);

        std::cout << transformed.getNbPoints() << std::endl;

        // Merge point clouds to map
        map.concatenate(transformed);
        // std::cout << map.getNbPoints() << std::endl;
        map = quadTreeSubsample->filter(map);
        // std::cout << map.getNbPoints() << std::endl;
        // Log current pose to csv file
        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        int64 time2 = std::stoll(parts[0]);
        ofs << T(0, 0) << "," << T(0, 1) << "," << T(0, 2) << "," << T(1, 0) << "," << T(1, 1) << "," << T(1, 2) << ",";  // NOLINT
        ofs << T(2, 0) << "," << T(2, 1) << "," << T(2, 2) << ",";
        ofs << time1 << "," << time2 << "\n";
    }
    // std::cout << map.getNbPoints() << std::endl;
    // map = quadTreeSubsample->filter(map);
    // std::cout << map.getNbPoints() << std::endl;

    icptime /= (radar_files.size() - 1);
    std::cout << "average ICP time: " << icptime << std::endl;
    map.save(map_name);

    return 0;
}
