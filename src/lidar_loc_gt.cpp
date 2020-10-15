#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "pointmatcher/PointMatcher.h"

typedef PointMatcher<double> PM;
typedef PM::DataPoints DP;

// ICP
Eigen::MatrixXd computeAndGetTransform2(Eigen::MatrixXd p2, Eigen::MatrixXd p1, double ransac_threshold,
    double inlier_ratio, int max_iterations) {
    Eigen::MatrixXd c1 = Eigen::MatrixXd::Ones(4, p1.cols());
    Eigen::MatrixXd c2 = Eigen::MatrixXd::Ones(4, p2.cols());
    c1.block(0, 0, 3, p1.cols()) = p1.block(0, 0, 3, p1.cols());
    c2.block(0, 0, 3, p2.cols()) = p2.block(0, 0, 3, p2.cols());
    std::cout << c1.cols() << " " << c1.rows() << std::endl;
    std::cout << c2.cols() << " " << c2.rows() << std::endl;
    DP::Labels labels;
    labels.push_back(DP::Label("x", 1));
    labels.push_back(DP::Label("y", 1));
    labels.push_back(DP::Label("z", 1));
    labels.push_back(DP::Label("w", 1));
    DP ref(c1, labels);
    DP data(c2, labels);
    PM::ICP icp;
    std::string config = "/home/keenan/radar_ws/src/yeti/yeti/config/icp_lidar_loc.yaml";
    std::ifstream ifs(config.c_str());
    icp.loadFromYaml(ifs);
    icp.readingDataPointsFilters = icp.referenceDataPointsFilters;
    // icp.setDefault();
    Eigen::Matrix4d prior = Eigen::Matrix4d::Identity();
    prior.block(0, 0, 2, 2) << -1, 0, 0, -1;
    PM::TransformationParameters T = icp(data, ref, prior);
    Eigen::MatrixXd Tout = T;

    // DP data_out(data);
    // icp.transformations.apply(data_out, T);
    // ref.save("test_ref.vtk");
    // data.save("test_data_in.vtk");
    // data_out.save("test_data_out.vtk");

    return Tout;
}

double getRotation(Eigen::MatrixXd T) {
    Eigen::MatrixXd Cmin = T.block(0, 0, 2, 2);
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(3, 3);
    C.block(0, 0, 2, 2) = Cmin;
    double trace = 0;
    for (int i = 0; i < C.rows(); ++i) {
        trace += C(i, i);
    }
    double phi = acos((trace - 1) / 2);

    if (T(0, 1) > 0)
        phi *= -1;

    return phi;
}

std::string get_lidar_file(std::vector<std::string> files, int64_t timestamp) {
    double minD = 0.15;
    int closest = -1;
    for (uint i = 0; i < files.size(); ++i) {
        std::vector<std::string> parts;
        boost::split(parts, files[i], boost::is_any_of("."));
        int64_t ltime = std::stoll(parts[0]);
        double diff = (timestamp - ltime) / 1.0e9;
        diff = fabs(diff);
        if (diff < minD) {
            minD = diff;
            closest = i;
        }
    }
    assert(closest >= 0);
    return files[closest];
}

int main(int argc, char *argv[]) {
    std::string root = "/home/keenan/Documents/data/boreas/2020_10_06";
    std::string datadir = root + "/lidar";
    std::string gt = root + "/lidar_files.csv";
    std::vector<std::string> lidar_files;
    get_file_names(datadir, lidar_files, "txt");

    // File for storing the results of estimation on each frame (and the accuracy)
    std::ofstream ofs;
    ofs.open("lidar_loc_accuracy.csv", std::ios::out);

    std::ifstream ifs(gt);
    std::string line;

    int i = 0;
    while (std::getline(ifs, line)) {
        std::cout << i << std::endl;
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        int64_t time1 = std::stoll(parts[0]);
        int64_t time2 = std::stoll(parts[1]);
        std::cout << time1 << " " << time2 << std::endl;

        Eigen::MatrixXd pc1;
        std::string lidar_file1 = get_lidar_file(lidar_files, time1);
        std::cout << lidar_file1 << std::endl;
        load_velodyne2(datadir + "/" + lidar_file1, pc1);
        std::cout << pc1.rows() << " " << pc1.cols() << std::endl;

        Eigen::MatrixXd pc2;
        std::string lidar_file2 = get_lidar_file(lidar_files, time2);
        // std::string lidar_file2 = "1602034313098256000.txt";
        std::cout << lidar_file2 << std::endl;
        load_velodyne2(datadir + "/" + lidar_file2, pc2);
        std::cout << pc2.rows() << " " << pc2.cols() << std::endl;

        Eigen::MatrixXd T = computeAndGetTransform2(pc2, pc1, 0, 0, 0);
        // T = T.inverse();

        std::cout << T << std::endl;

        // Retrieve the ground truth to calculate accuracy
        float yaw = getRotation(T);

        // Write estimated and ground truth transform to the csv file
        ofs << time1 << "," << time2 << "," << T(0, 3) << "," << T(1, 3) << "," << yaw << "\n";
        i++;
    }

    return 0;
}
