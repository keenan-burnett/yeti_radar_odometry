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

double beta = 0.0488;

Eigen::MatrixXd get_transform(std::vector<double> gt) {
    Eigen::Vector3d p = {gt[1], gt[2], gt[3]};
    Eigen::Vector4d qbar = {gt[4], gt[5], gt[6], gt[7]};
    double EPS = 1e-15;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3);
    if (qbar.transpose() * qbar > EPS) {
        Eigen::Vector3d epsilon = {gt[4], gt[5], gt[6]};
        double eta = gt[7];
        Eigen::Matrix3d epsilon_cross;
        epsilon_cross << 0, -epsilon(2), epsilon(1),
                         epsilon(2), 0, -epsilon(0),
                         -epsilon(1), epsilon(0), 0;
        Eigen::Matrix3d I = Eigen::MatrixXd::Identity(3, 3);
        R = (pow(eta, 2.0) - epsilon.transpose() * epsilon) * I +
            2 * (epsilon * epsilon.transpose()) - 2 * eta * epsilon_cross;
    }
    enforce_orthogonality(R);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T << R.transpose(), p, 0, 0, 0, 1.0;
    return T;
}

void undistort_radar_image(cv::Mat &input, cv::Mat &output, Eigen::VectorXd wbar, float cart_resolution,
    int cart_pixel_width) {

    std::vector<double> azimuths(400);
    azimuths[0] = 0;
    double azimuth_step = M_PI / 200;
    for (uint i = 1; i < azimuths.size(); ++i) {
        azimuths[i] = azimuths[i - 1] + azimuth_step;
    }
    std::vector<double> times(400);
    times[0] = 0;
    double time_step = 0.000625;
    for (uint i = 1; i < times.size(); ++i) {
        times[i] = times[i - 1] + time_step;
    }
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;

    cv::Mat map_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat map_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < map_y.cols; ++j) {
        for (int i = 0; i < map_y.rows; ++i) {
            map_y.at<float>(i, j) = -1 * cart_min_range + j * cart_resolution;
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < map_x.rows; ++i) {
        for (int j = 0; j < map_x.cols; ++j) {
            map_x.at<float>(i, j) = cart_min_range - i * cart_resolution;
        }
    }

    std::vector<Eigen::MatrixXd> transforms(400);
    transforms[0] = Eigen::MatrixXd::Identity(4, 4);

#pragma omp parallel for
    for (uint i = 1; i < transforms.size(); ++i) {
        Eigen::MatrixXd T = se3ToSE3(wbar * times[i]);
        transforms[i] = T.inverse();
    }

    cv::Mat orig_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat orig_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    double vel = fabs(wbar(0, 0));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < orig_x.rows; ++i) {
        for (int j = 0; j < orig_y.cols; ++j) {
            float x = map_x.at<float>(i, j);
            float y = map_y.at<float>(i, j);

            // Doppler distortion
            double rsq = x * x + y * y;
            x -= beta * vel * x * x / rsq;
            y -= beta * vel * x * y / rsq;

            Eigen::Vector4d pbar = {x, y, 0, 1};

            // Motion distortion
            float phi = atan2f(y, x);
            if (phi < 0)
                phi += 2 * M_PI;
            int row = (phi - azimuths[0]) / azimuth_step;
            pbar = transforms[row] * pbar;

            float u = (cart_min_range + pbar(1)) / cart_resolution;
            float v = (cart_min_range - pbar(0)) / cart_resolution;
            orig_x.at<float>(i, j) = u;
            orig_y.at<float>(i, j) = v;
        }
    }

    cv::remap(input, output, orig_x, orig_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

int main() {
    std::string root = "/home/keenan/Documents/data/boreas/2020_09_01/1598986495";
    std::string radardir = root + "/radar";
    std::string lidardir = root + "/lidar";
    std::string gt = root + "/gps.csv";
    float cart_resolution = 0.2384;
    int cart_pixel_width = 586;
    int min_range = 42;
    float radar_resolution = 0.0596;
    bool interp = true;
    float zq = 3.0;
    int sigma_gauss = 17;
    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(radardir, radar_files);
    std::vector<std::string> lidar_files;
    get_file_names(lidardir, lidar_files, "txt");
    // Get transform from LIDAR to radar frame
    Eigen::Matrix4d T_radar_lidar = Eigen::Matrix4d::Identity();
    double rot = 0.0483;
    T_radar_lidar.block(0, 0, 2, 2) << cos(rot), sin(rot), -sin(rot), cos(rot);

    cv::Mat f1;
    std::vector<int64_t> t1;

    for (uint i = 27; i < radar_files.size(); ++i) {
        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        time1 -= 0.10 * 1e9;
        std::cout << "time: " << time1 << std::endl;

        // Get gps/imu info for the radar scan
        std::vector<double> gtvec;
        assert(get_groundtruth_odometry2(gt, time1, gtvec));

        Eigen::Matrix4d T_radar = get_transform(gtvec);

        double velocity = sqrt(pow(gtvec[8], 2) + pow(gtvec[9], 2));
        double omega = gtvec[gtvec.size() - 1];
        std::cout << " v: " << velocity << " w: " << omega << std::endl;
        Eigen::VectorXd wbar = Eigen::VectorXd::Zero(6);
        wbar(0) = velocity;
        wbar(5) = omega;

        std::vector<int64_t> times;
        std::vector<double> azimuths;
        std::vector<bool> valid;
        Eigen::MatrixXd targets;
        Eigen::MatrixXd cart_targets;
        std::vector<cv::Point2f> bev_points;
        load_radar(radardir + "/" + radar_files[i], times, azimuths, valid, f1, CIR204);

        // Get LIDAR pointcloud with timestamp closest to this radar scan
        int64_t min_delta = 1000000000000;
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
        std::cout << "min_delta: " << double(min_delta / 1.0e9) << std::endl;

        std::vector<int64_t> lidar_times;
        std::vector<double> lidar_azimuths;
        Eigen::MatrixXd pc;
        load_velodyne2(lidardir + "/" + lidar_files[closest_lidar], pc);

        // Need to transform pc into radar frame, and then remove motion distortion
        Eigen::MatrixXd pc_distort = Eigen::MatrixXd::Zero(2, pc.cols());
        Eigen::MatrixXd T_motion = Eigen::MatrixXd::Identity(4, 4);
        for (uint j = 0; j < pc.cols(); ++j) {
            if (pc(2, j) < -1.4)
                continue;

            if (sqrt(pow(pc(0, j), 2) + pow(pc(1, j), 2)) < 2.0)
                continue;

            // Remove motion distortion from LIDAR data
            // double phi = atan2f(pc(1, j), pc(0, j));
            // if (phi < 0)
            //     phi += 2 * M_PI;
            // // phi = 2 * M_PI - phi;
            // double timestamp = 0.1 * (phi / (2 * M_PI));
            // T_motion = se3ToSE3(wbar * timestamp);

            Eigen::Vector4d p1bar = {pc(0, j), pc(1, j), 0, 1};
            p1bar = T_radar_lidar * T_motion * p1bar;
            pc_distort(0, j) = p1bar(0);
            pc_distort(1, j) = -p1bar(1);
        }

        // Convert radar FFT data into cartesian image
        cv::Mat cart_img;
        radar_polar_to_cartesian(azimuths, f1, radar_resolution, cart_resolution, cart_pixel_width, interp, cart_img,
            CV_32F, CIR204);

        double min, max;
        cv::minMaxLoc(cart_img, &min, &max);
        cart_img *= 1.5;

        cv::Mat vis;
        draw_points(cart_img, pc_distort, cart_resolution, cart_pixel_width, vis);
        Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(2, 4);
        double size = 2.5;
        cross << -size, size, 0, 0, 0, 0, -size, size;
        std::vector<cv::Point2f> cross_points;
        convert_to_bev(cross, cart_resolution, cart_pixel_width, cross_points);
        cv::line(vis, cross_points[0], cross_points[1], cv::Scalar(255, 255, 255));
        cv::line(vis, cross_points[2], cross_points[3], cv::Scalar(255, 255, 255));

        cv::imshow("distorted", vis);

        // Remove distortion from the radar cartesian image
        cv::Mat undistort;
        undistort_radar_image(cart_img, undistort, wbar, cart_resolution, cart_pixel_width);
        cv::Mat vis2;
        draw_points(undistort, pc_distort, cart_resolution, cart_pixel_width, vis2);
        cv::line(vis2, cross_points[0], cross_points[1], cv::Scalar(255, 255, 255));
        cv::line(vis2, cross_points[2], cross_points[3], cv::Scalar(255, 255, 255));
        int buffer = 2;
        cv::ellipse(vis2, cv::Point(280, 177), cv::Size(11 + buffer, 40 + buffer), 0, 0, 360, cv::Scalar(255, 0, 0), 2);
        cv::ellipse(vis2, cv::Point(270, 316), cv::Size(12 + buffer, 36 + buffer), 0, 0, 360, cv::Scalar(255, 0, 0), 2);
        cv::ellipse(vis2, cv::Point(274, 384), cv::Size(11 + buffer, 34 + buffer), 0, 0, 360, cv::Scalar(255, 0, 0), 2);
        cv::ellipse(vis2, cv::Point(256, 449), cv::Size(14 + buffer, 41 + buffer), 0, 0, 360, cv::Scalar(255, 0, 0), 2);

        cv::imshow("motion distort removed", vis2);
        cv::waitKey(0);
    }
}
