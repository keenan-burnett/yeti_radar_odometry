#include <yaml-cpp/yaml.h>
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

int main(int argc, char *argv[]) {
    std::string root = "/home/keenan/Documents/data/";
    // std::string root = "/workspace/raid/krb/oxford-radar-robotcar-dataset/";
    std::string sequence = "2019-01-10-14-36-48-radar-oxford-10k-partial";
    if (argc > 1)
        sequence = argv[1];
    std::string append = "";
    if (argc > 2)
        append = argv[2];
    std::cout << sequence << std::endl;
    std::cout << append << std::endl;
    omp_set_num_threads(8);
    std::string datadir = root + sequence + "/radar";
    std::string gt = root + sequence + "/gt/radar_odometry.csv";
    YAML::Node node = YAML::LoadFile("/home/keenan/radar_ws/src/yeti/yeti/config/feature_matching.yaml");
    // YAML::Node node = YAML::LoadFile("/workspace/Documents/catkin_ws/src/yeti/yeti/config/feature_matching.yaml");

    float cart_resolution = node["cart_resolution"].as<float>();
    int cart_pixel_width = node["cart_pixel_width"].as<int>();
    int min_range = node["min_range"].as<int>();
    float radar_resolution = node["radar_resolution"].as<float>();
    bool interp = node["interp"].as<bool>();
    float zq = node["zq"].as<float>();
    int max_points = node["max_points"].as<int>();
    int sigma_gauss = node["sigma_gauss"].as<int>();
    int patch_size = node["patch_size"].as<int>();
    float nndr = node["nndr"].as<float>();
    double ransac_threshold = node["threshold"].as<double>();
    double inlier_ratio = node["inlier_ratio"].as<double>();
    int max_iterations = node["max_iterations"].as<int>();
    int max_gn_iterations = node["max_gn_iterations"].as<int>();
    double md_threshold = node["md_threshold"].as<double>();
    int keypoint_extraction = node["keypoint_extraction"].as<int>();
    double beta = node["beta"].as<double>();

    float cart_res2 = 0.3456;
    int cart_width2 = 722;

    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);
    // File for storing the results of estimation on each frame (and the accuracy)
    std::ofstream ofs;
    ofs.open("accuracy" + append + ".csv", std::ios::out);
    ofs << "x,y,yaw,gtx,gty,gtyaw,time1,time2,xmd,ymd,yawmd,xdopp,ydopp,yawdopp\n";
    // Create ORB feature detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    // BRUTEFORCE_HAMMING for ORB descriptors
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    cv::Mat img1, img2, desc1, desc2;
    std::vector<cv::KeyPoint> kp1, kp2;
    Eigen::MatrixXd targets, cart_targets1, cart_targets2;
    std::vector<int64_t> t1, t2;

    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    for (uint i = 0; i < radar_files.size() - 1; ++i) {
        std::cout << i << "/" << radar_files.size() << std::endl;
        if (i > 0) {
            t1 = t2; desc1 = desc2.clone(); cart_targets1 = cart_targets2;
            kp1 = kp2; img2.copyTo(img1);
        }
        load_radar(datadir + "/" + radar_files[i], times, azimuths, valid, fft_data);
        if (keypoint_extraction == 0)
            cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
        if (keypoint_extraction == 1)
            cen2019features(fft_data, max_points, min_range, targets);
        if (keypoint_extraction == 0 || keypoint_extraction == 1) {
            radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp, img2, CV_8UC1);  // NOLINT
            polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets2, t2);
            convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, patch_size, kp2, t2);
            cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
                radar_resolution, cart_res2, cart_width2, desc2);
            // detector->compute(img2, kp2, desc2);
        }
        if (keypoint_extraction == 2) {
            detector->detect(img2, kp2);
            detector->compute(img2, kp2, desc2);
            convert_from_bev(kp2, cart_resolution, cart_pixel_width, cart_targets2);
            getTimes(cart_targets2, azimuths, times, t2);
        }
        if (i == 0)
            continue;
        // Match keypoint descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        auto start = std::chrono::high_resolution_clock::now();
        matcher->knnMatch(desc1, desc2, knn_matches, 2);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e = stop - start;
        std::cout << "match: " << e.count() << std::endl;

        // Filter matches using nearest neighbor distance ratio (Lowe, Szeliski)
        std::vector<cv::DMatch> good_matches;
        for (uint j = 0; j < knn_matches.size(); ++j) {
            if (!knn_matches[j].size())
                continue;
            if (knn_matches[j][0].distance < nndr * knn_matches[j][1].distance) {
                good_matches.push_back(knn_matches[j][0]);
            }
        }
        // Convert the good key point matches to Eigen matrices
        Eigen::MatrixXd p1 = Eigen::MatrixXd::Zero(2, good_matches.size());
        Eigen::MatrixXd p2 = p1;
        std::vector<int64_t> t1prime = t1, t2prime = t2;
        for (uint j = 0; j < good_matches.size(); ++j) {
            p1(0, j) = cart_targets1(0, good_matches[j].queryIdx);
            p1(1, j) = cart_targets1(1, good_matches[j].queryIdx);
            p2(0, j) = cart_targets2(0, good_matches[j].trainIdx);
            p2(1, j) = cart_targets2(1, good_matches[j].trainIdx);
            t1prime[j] = t1[good_matches[j].queryIdx];
            t2prime[j] = t2[good_matches[j].trainIdx];
        }
        t1prime.resize(good_matches.size());
        t2prime.resize(good_matches.size());

        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        int64 time2 = std::stoll(parts[0]);
        double delta_t = (time2 - time1) / 1000000.0;

        // Compute the transformation using RANSAC
        Ransac ransac(p2, p1, ransac_threshold, inlier_ratio, max_iterations);
        srand(i);
        ransac.computeModel();
        Eigen::MatrixXd T;  // T_1_2
        ransac.getTransform(T);

        std::vector<int> inliers;
        ransac.getInliers(T, inliers);
        std::cout << "rigid inliers: " << inliers.size() << std::endl;

        // Compute the transformation using motion-distorted RANSAC
        MotionDistortedRansac mdransac(p2, p1, t2prime, t1prime, md_threshold, inlier_ratio, max_iterations);
        mdransac.setMaxGNIterations(max_gn_iterations);
        mdransac.correctForDoppler(false);
        srand(i);
        mdransac.computeModel();
        Eigen::MatrixXd Tmd;
        mdransac.getTransform(delta_t, Tmd);
        Tmd = Tmd.inverse();

        Eigen::VectorXd wbar;
        mdransac.getMotion(wbar);
        inliers.clear();
        mdransac.getInliers(wbar, inliers);
        std::cout << "mdransac inliers: " << inliers.size() << std::endl;

        // MDRANSAC + Doppler
        mdransac.correctForDoppler(true);
        mdransac.setDopplerParameter(beta);
        srand(i);
        mdransac.computeModel();
        Eigen::MatrixXd Tmd2 = Eigen::MatrixXd::Zero(4, 4);
        mdransac.getTransform(delta_t, Tmd2);
        Tmd2 = Tmd2.inverse();

        // Retrieve the ground truth to calculate accuracy
        std::vector<float> gtvec;
        if (!get_groundtruth_odometry(gt, time1, time2, gtvec)) {
            std::cout << "ground truth odometry for " << time1 << " " << time2 << " not found... exiting." << std::endl;
            return 0;
        }
        float yaw = -1 * asin(T(0, 1));
        float yaw2 = -1 * asin(Tmd(0, 1));
        float yaw3 = -1 * asin(Tmd2(0, 1));

        // Write estimated and ground truth transform to the csv file
        ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ",";
        ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[5] << ",";
        ofs << time1 << "," << time2 << "," << Tmd(0, 3) << "," << Tmd(1, 3) << "," <<  yaw2 << ",";
        ofs << Tmd2(0, 3) << "," << Tmd2(1, 3) << "," << yaw3 << "\n";

        // cv::Mat img_matches;
        // cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, cv::Scalar::all(-1),
        //          cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // cv::imshow("good", img_matches);
        // cv::waitKey(0);
    }

    return 0;
}
