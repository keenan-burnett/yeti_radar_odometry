#include <iostream>
#include <string>
#include <fstream>
#include "matplotlibcpp.h"  // NOLINT
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "association.hpp"
namespace plt = matplotlibcpp;

Eigen::Matrix4d get_transform(std::vector<float> t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    double theta = t[5];
    T(0, 0) = cos(theta); T(0, 1) = -sin(theta);
    T(1, 0) = sin(theta); T(1, 1) = cos(theta);
    T(0, 3) = t[0];
    T(1, 3) = t[1];
    return T;
}

Eigen::Matrix4d get_transform_from_file(std::string path) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    double x, y, z, r, p, yaw;
    std::ifstream ifs;
    ifs.open(path);
    while (ifs >> x >> y >> z >> r >> p >> yaw) {
        Eigen::MatrixXd C = Eigen::Matrix3d::Identity(3, 3);
        Eigen::MatrixXd C1 = C, C2 = C, C3 = C;
        C1 << 1, 0, 0, 0, cos(r), -sin(r), 0, -sin(r), cos(r);
        C2 << cos(p), 0, sin(p), 0, 1, 0, -sin(p), 0, cos(p);
        C3 << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1;
        C = C3 * C2 * C1;
        enforce_orthogonality(C);
        T.block(0, 0, 3, 3) = C;
        T(0, 3) = x;
        T(1, 3) = y;
        T(2, 3) = z;
    }
    return T;
}

int main() {
    std::string root = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial";
    std::string radardir = root + "/radar";
    std::string lidardir = root + "/velodyne_right";
    std::string gt = root + "/gt/radar_odometry.csv";
    float cart_resolution = 0.25;
    int cart_pixel_width = 1000;
    int min_range = 58;
    float radar_resolution = 0.0432;
    bool interp = true;
    float zq = 3.0;
    int max_points = 10000;
    int sigma_gauss = 17;
    int patch_size = 21;
    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(radardir, radar_files);
    std::vector<std::string> lidar_files;
    get_file_names(lidardir, lidar_files, "png");
    // Get transform from LIDAR to radar frame
    Eigen::Matrix4d T_stereo_lidar = get_transform_from_file("/home/keenan/Documents/git/robotcar-dataset-sdk/extrinsics/velodyne_right.txt"); // NOLINT
    Eigen::Matrix4d T_stereo_radar = get_transform_from_file("/home/keenan/Documents/git/robotcar-dataset-sdk/extrinsics/radar.txt"); // NOLINT
    Eigen::Matrix4d T_radar_lidar = T_stereo_radar.inverse() * T_stereo_lidar;

    cv::Mat f1;
    std::vector<int64_t> t1;

    for (uint i = 31; i < radar_files.size(); ++i) {
        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        int64 time2 = std::stoll(parts[0]);

        std::vector<float> gtvec;
        get_groundtruth_odometry(gt, time1, time2, gtvec);
        Eigen::Matrix4d T_gt = get_transform(gtvec);  // T21

        Eigen::VectorXd w_gt = SE3tose3(T_gt);
        double delta_t = (time2 - time1) / 1000000.0;
        w_gt /= delta_t;

        // if (fabs(w_gt(0)) < 2.0)
            // continue;

        std::cout << "radar file: " << i << std::endl;
        std::cout << "time: " << time1 << std::endl;

        std::cout << T_gt << std::endl;
        std::cout << w_gt << std::endl;

        std::vector<int64_t> times;
        std::vector<double> azimuths;
        std::vector<bool> valid;
        Eigen::MatrixXd targets;
        Eigen::MatrixXd cart_targets;
        std::vector<cv::Point2f> bev_points;
        load_radar(radardir + "/" + radar_files[i], times, azimuths, valid, f1);
        cen2019features(f1, max_points, min_range, targets);
        // cen2018features(f1, zq, sigma_gauss, min_range, targets);
        polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets, t1);

        std::vector<double> x2, y2;
        for (uint j = 1; j < t1.size(); ++j) {
            double delta_t = (t1[j] - t1[0])/1000000.0;
            Eigen::MatrixXd T = se3ToSE3(w_gt * delta_t);
            Eigen::Vector4d p1bar = {cart_targets(0, j), cart_targets(1, j), 0, 1};
            p1bar = T.inverse() * p1bar;
            x2.push_back(p1bar(0));
            y2.push_back(p1bar(1));
        }

        // Get LIDAR pointcloud with timestamp closest to this radar scan
        int64_t min_delta = 1000000000;
        int closest_lidar = 0;
        for (uint j = 0; j < lidar_files.size(); ++j) {
            std::vector<std::string> parts;
            boost::split(parts, lidar_files[j], boost::is_any_of("."));
            int64 t = std::stoll(parts[0]);
            if (abs(t - time1) < min_delta) {
                min_delta = abs(t - time1);
                closest_lidar = j;
            }
        }
        std::vector<int64_t> lidar_times;
        std::vector<double> lidar_azimuths;
        Eigen::MatrixXd pc;
        load_velodyne(lidardir + "/" + lidar_files[closest_lidar], lidar_times, lidar_azimuths, pc);
        // Need to transform pc into radar frame, and then remove motion distortion
        std::vector<double> x3, y3;
        for (uint j = 0; j < pc.cols(); ++j) {
            if (pc(2, j) > 1.2)
                continue;
            double delta_t = 0.000046296 * j;
            Eigen::MatrixXd T = se3ToSE3(w_gt * delta_t);
            Eigen::Vector4d p1bar = {pc(0, j), pc(1, j), 0, 1};
            p1bar = T_radar_lidar * T.inverse() * p1bar;
            x3.push_back(p1bar(0));
            y3.push_back(p1bar(1));
        }

        std::map<std::string, std::string> kw3;
        kw3.insert(std::pair<std::string, std::string>("c", "r"));
        plt::scatter(x3, y3, 25.0, kw3);
        std::map<std::string, std::string> kw2;
        kw2.insert(std::pair<std::string, std::string>("c", "b"));
        plt::scatter(x2, y2, 25.0, kw2);
        plt::show();
    }
    // Extract radar keypoints

    // Only keep one point per azimuth?
}
