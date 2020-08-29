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
#include "association.hpp"

void getTimes(Eigen::MatrixXd cart_targets, std::vector<double> azimuths, std::vector<int64_t> times,
    std::vector<int64_t> &tout) {
    tout.clear();
    for (uint j = 0; j < cart_targets.cols(); ++j) {
        double theta = wrapto2pi(atan2(cart_targets(1, j), cart_targets(0, j)));
        double closest = 0;
        double mindiff = 1000;
        for (uint k = 0; k < mindiff; ++k) {
            if (fabs(theta - azimuths[k]) < mindiff) {
                mindiff = fabs(theta - azimuths[k]);
                closest = k;
            }
            tout.push_back(times[closest]);
        }
    }
}

struct lesser_than_d {
    inline bool operator() (const cv::DMatch& d1, const cv::DMatch& d2) {
        return d1.queryIdx < d2.queryIdx;
    }
};

int main(int argc, char *argv[]) {
    std::string root = "/home/keenan/Documents/data/";
    std::string sequence = "2019-01-10-14-36-48-radar-oxford-10k-partial";
    if (argc > 1)
        sequence = argv[1];
    std::string datadir = root + sequence + "/radar";
    std::string gt = root + sequence + "/gt/radar_odometry.csv";
    float cart_resolution = 0.25;
    int cart_pixel_width = 1000;
    int min_range = 58;
    float radar_resolution = 0.0432;
    bool interp = true;
    float zq = 3.0;
    // int max_points = 10000;
    int sigma_gauss = 17;
    int patch_size = 27;
    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);
    // File for storing the results of estimation on each frame (and the accuracy)
    std::string accfile = "accuracy.csv";
    std::ofstream ofs;
    ofs.open(accfile, std::ios::out);
    ofs << "x,y,yaw,gtx,gty,gtyaw,time1,time2,xmd,ymd,yawmd\n";
    // Create ORB feature detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    detector->setMaxFeatures(1000);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    cv::Mat img1, img2, desc1, desc2;
    std::vector<cv::KeyPoint> kp1, kp2;
    Eigen::MatrixXd targets, cart_targets1, cart_targets2;
    std::vector<int64_t> t1, t2;

    // double ransactime = 0;
    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    for (uint i = 0; i < radar_files.size() - 1; ++i) {
        std::cout << i << std::endl;

        if (i == 0) {
            load_radar(datadir + "/" + radar_files[i], times, azimuths, valid, fft_data);
            cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
            // cen2019features(f1, max_points, min_range, targets);
            polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets1, t1);
            convert_to_bev(cart_targets1, cart_resolution, cart_pixel_width, patch_size, kp1, t1);
            radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp, img1, CV_8UC1);  // NOLINT

            // detector->detect(img1, kp1);
            detector->compute(img1, kp1, desc1);
            // convert_from_bev(kp1, cart_resolution, cart_pixel_width, cart_targets1);
            // getTimes(cart_targets1, azimuths, times, t1);

        } else {
            t1 = t2; desc1 = desc2.clone(); cart_targets1 = cart_targets2;
            kp1 = kp2; img2.copyTo(img1);
        }
        load_radar(datadir + "/" + radar_files[i + 1], times, azimuths, valid, fft_data);
        cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
        // std::cout << "targets: " << targets.cols() << std::endl;
        // cen2019features(f2, max_points, min_range, targets);
        polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets2, t2);
        // std::cout << "cart_targets2: " << cart_targets2.cols() << std::endl;
        convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, patch_size, kp2, t2);
        // std::cout << "kp2: " << kp2.size() << std::endl;
        radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp, img2, CV_8UC1);  // NOLINT

        // detector->detect(img2, kp2);
        detector->compute(img2, kp2, desc2);
        // std::cout << "kp2: " << kp2.size() << std::endl;
        // convert_from_bev(kp2, cart_resolution, cart_pixel_width, cart_targets2);
        // getTimes(cart_targets2, azimuths, times, t2);

        // Match keypoint descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(desc1, desc2, knn_matches, 2);
        // Filter matches using nearest neighbor distance ratio (Lowe, Szeliski)
        const float ratio = 0.7;
        std::vector<cv::DMatch> good_matches;
        for (uint j = 0; j < knn_matches.size(); ++j) {
            if (!knn_matches[j].size())
                continue;
            if (knn_matches[j][0].distance < ratio * knn_matches[j][1].distance) {
                good_matches.push_back(knn_matches[j][0]);
            }
        }
        // std::cout << "good matches: " << good_matches.size() << std::endl;

        // Convert the good key point matches to Eigen matrices
        Eigen::MatrixXd p1 = Eigen::MatrixXd::Zero(2, good_matches.size());
        Eigen::MatrixXd p2 = p1;
        std::vector<int64_t> t1prime = t1, t2prime = t2;
        int64_t check1 = 0, check2 = 0;
        for (uint j = 0; j < good_matches.size(); ++j) {
            p1(0, j) = cart_targets1(0, good_matches[j].queryIdx);
            p1(1, j) = cart_targets1(1, good_matches[j].queryIdx);
            p2(0, j) = cart_targets2(0, good_matches[j].trainIdx);
            p2(1, j) = cart_targets2(1, good_matches[j].trainIdx);
            t1prime[j] = t1[good_matches[j].queryIdx];
            check1 += t1prime[j];
            t2prime[j] = t2[good_matches[j].trainIdx];
            check2 += t2prime[j];
        }
        t1prime.resize(good_matches.size());
        t2prime.resize(good_matches.size());
        // std::cout << "p1sum: " << p1.sum() << std::endl;
        // std::cout << "p2sum: " << p2.sum() << std::endl;
        // std::cout << "check1: " << check1 << std::endl;
        // std::cout << "check2: " << check2 << std::endl;
        // std::cout << p1 << std::endl;
        // std::cout << p2 << std::endl;

        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        int64 time2 = std::stoll(parts[0]);
        double delta_t = (time2 - time1) / 1000000.0;

        // Compute the transformation using RANSAC
        Ransac ransac(p2, p1, 0.35, 0.90, 100);
        ransac.computeModel();
        Eigen::MatrixXd T;
        ransac.getTransform(T);
        Eigen::MatrixXd T2 = Eigen::MatrixXd::Identity(4, 4);
        T2.block(0, 0, 2, 2) = T.block(0, 0, 2, 2);
        T2.block(0, 3, 2, 1) = T.block(0, 2, 2, 1);

        auto start = std::chrono::high_resolution_clock::now();
        MotionDistortedRansac mdransac(p2, p1, t2prime, t1prime, 0.35 * 0.35, 0.90, 100);
        mdransac.setMaxGNIterations(10);
        mdransac.computeModel();
        Eigen::VectorXd w;
        mdransac.getMotion(w);
        Eigen::MatrixXd Tmd;
        mdransac.getTransform(delta_t, Tmd);
        Tmd = Tmd.inverse();
        w *= -1;

        // Compute the transformation using motion-distorted RANSAC
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e = stop - start;
        std::cout << "ransac: " << e.count() << std::endl;

        // Retrieve the ground truth to calculate accuracy
        std::vector<float> gtvec;
        get_groundtruth_odometry(gt, time1, time2, gtvec);
        float yaw = -1 * asin(T(0, 1));
        float yaw2 = -1 * asin(Tmd(0, 1));
        // Write estimated and ground truth transform to the csv file
        ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ",";
        ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[5] << ",";
        ofs << time1 << "," << time2 << "," << Tmd(0, 3) << "," << Tmd(1, 3) << "," <<  yaw2 << "\n";
        // cv::Mat img_matches;
        // cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, cv::Scalar::all(-1),
        //          cv::Scalar::all(-1), std::vector<char>());
        // cv::imshow("good", img_matches);
        // cv::waitKey(0);
    }
    return 0;
}
