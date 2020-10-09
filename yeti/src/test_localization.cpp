#include <yaml-cpp/yaml.h>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <nanoflann.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "association.hpp"

typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf> my_kd_tree_t;

int main(int argc, char *argv[]) {
    std::string root = "/home/keenan/Documents/data/boreas/2020_10_06";
    omp_set_num_threads(8);
    std::string datadir = root + "/radar";
    std::string gt = root + "/radar_groundtruth.csv";
    YAML::Node node = YAML::LoadFile("/home/keenan/radar_ws/src/yeti/yeti/config/feature_matching.yaml");

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
    double beta = node["beta"].as<double>();

    float cart_resolution = 0.2384;
    int cart_pixel_width = 1048;
    int min_range = 42;
    float radar_resolution = 0.0596;

    float cart_res2 = 0.3576;
    int cart_width2 = 700;

    // File for storing the results of estimation on each frame (and the accuracy)
    std::ofstream ofs;
    ofs.open("localization_accuracy.csv", std::ios::out);
    ofs << "x,y,yaw,gtx,gty,gtyaw,time1,time2,xmd,ymd,yawmd,xdopp,ydopp,yawdopp,v1,w1,v2,w2,\n";
    // Create ORB feature detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);

    cv::Mat img1, img2, desc1, desc2;
    std::vector<cv::KeyPoint> kp1, kp2;
    Eigen::MatrixXd targets, cart_targets1, cart_targets2;
    std::vector<int64_t> t1, t2;

    std::vector<int64_t> times;
    std::vector<double> azimuths;
    std::vector<bool> valid;
    cv::Mat fft_data;

    std::ifstream ifs(gt);
    std::string line;

    int i = 0;
    while (std::getline(ifs, line)) {
        std::cout << i << std::endl;
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        int64_t time1 = std::stoll(parts[0]) / 1000;
        int64_t time2 = std::stoll(parts[1]) / 1000;
        std::vector<double> gtvec;
        for (uint j = 2; j < 9; ++j) {
            gtvec.push_back(std::stod(parts[j]));
        }

        std::string radar_file1 = parts[0] + ".png";

        load_radar(datadir + "/" + radar_file1, times, azimuths, valid, fft_data, CIR204);
        cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
        radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp,
            img1, CV_8UC1, CIR204);
        polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets1, t1);
        convert_to_bev(cart_targets1, cart_resolution, cart_pixel_width, patch_size, kp1, t1);
        // detector->compute(img1, kp1, desc1);
        cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets1,
            radar_resolution, cart_res2, cart_width2, desc1);

        std::string radar_file2 = parts[1] + ".png";

        load_radar(datadir + "/" + radar_file2, times, azimuths, valid, fft_data, CIR204);
        cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
        radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp, img2,
            CV_8UC1, CIR204);
        polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets2, t2);
        convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, patch_size, kp2, t2);
        // detector->compute(img2, kp2, desc2);
        cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
            radar_resolution, cart_res2, cart_width2, desc2);

        // Match keypoint descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(desc1, desc2, knn_matches, 2);

        // Eigen::MatrixXf d1, d2;
        // cv2eigen(desc1, d1);
        // cv2eigen(desc2, d2);
        //
        // d1 = Eigen::MatrixXf::Random(d1.rows(), 3);
        // d2 = Eigen::MatrixXf::Random(d2.rows(), 3);
        //
        // std::cout << "d1: " << d1.rows() << " " << d1.cols() << std::endl;
        // std::cout << "d2: " << d2.rows() << " " << d2.cols() << std::endl;
        //
        // // Eigen::Map<Eigen::MatrixXf> d1(desc1.ptr<float>(), desc1.rows, desc1.cols);
        // // Eigen::Map<Eigen::MatrixXf> d2(desc2.ptr<float>(), desc2.rows, desc2.cols);
        //
        // int dim = d2.cols();
        // my_kd_tree_t mat_index(dim, std::cref(d2), 10);
        // mat_index.index->buildIndex();
        //
        // std::cout << "built kdtree" << std::endl;
        //
        // const size_t num_results = 2;
        // std::vector<size_t> ret_indexes(num_results);
        // std::vector<float> out_dists_sqr(num_results);
        // nanoflann::KNNResultSet<float> resultSet(num_results);
        // resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        //
        // std::vector<std::vector<cv::DMatch>> knn_matches(d1.rows());
        // for (uint j = 0; j < d1.rows(); ++j) {
        //     std::vector<float> query_pt(dim);
        //     for (size_t d = 0; d < dim; ++d) {
        //         query_pt[d] = d1(j, d);
        //     }
        //     mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
        //     knn_matches[j].clear();
        //     knn_matches[j].push_back(cv::DMatch(j, ret_indexes[0], sqrt(out_dists_sqr[0])));
        //     knn_matches[j].push_back(cv::DMatch(j, ret_indexes[1], sqrt(out_dists_sqr[1])));
        //     std::cout << j << " " << ret_indexes[0] << " " << ret_indexes[1] << " " << out_dists_sqr[0] << " " << out_dists_sqr[1] << std::endl;
        // }
        //
        // std::cout << "survived" << std::endl;

        // std::vector<cv::DMatch> good_matches;
        // for (uint j = 0; j < cart_targets2.cols(); ++j) {
        //     float minD = 1.0e6;
        //     int best = -1;
        //     for (uint k = 0; k < cart_targets1.cols(); ++k) {
        //         float e = (d1.block(k, 0, 1, d) - d2.block(j, 0, 1, d)).squaredNorm();
        //         if (e < minD) {
        //             minD = e;
        //             best = k;
        //         }
        //     }
        //     if (best >= 0)
        //         good_matches.push_back(cv::DMatch(best, j, minD));
        // }
        //
        // std::cout << "finish" << std::endl;

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

        // double delta_t = (time2 - time1) / 1.0e6;
        double delta_t = 0.25;

        std::cout << time1 << " " << time2 << " " << delta_t << std::endl;

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
        // MotionDistortedRansac mdransac(p2, p1, t2prime, t1prime, md_threshold, inlier_ratio, max_iterations);
        // mdransac.setMaxGNIterations(max_gn_iterations);
        // mdransac.correctForDoppler(false);
        // srand(i);
        // mdransac.computeModel();
        Eigen::MatrixXd Tmd = Eigen::MatrixXd::Zero(4, 4);
        // mdransac.getTransform(delta_t, Tmd);
        // Tmd = Tmd.inverse();
        //
        // Eigen::VectorXd wbar;
        // mdransac.getMotion(wbar);
        // inliers.clear();
        // mdransac.getInliers(wbar, inliers);
        // std::cout << "mdransac inliers: " << inliers.size() << std::endl;
        //
        // // MDRANSAC + Doppler
        // mdransac.correctForDoppler(true);
        // mdransac.setDopplerParameter(beta);
        // srand(i);
        // mdransac.computeModel();
        Eigen::MatrixXd Tmd2 = Eigen::MatrixXd::Zero(4, 4);
        // mdransac.getTransform(delta_t, Tmd2);
        // Tmd2 = Tmd2.inverse();

        // Retrieve the ground truth to calculate accuracy
        float yaw = -1 * asin(T(0, 1));
        float yaw2 = -1 * asin(Tmd(0, 1));
        float yaw3 = -1 * asin(Tmd2(0, 1));

        // Write estimated and ground truth transform to the csv file
        ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ",";
        ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[2] << ",";
        ofs << time1 << "," << time2 << "," << Tmd(0, 3) << "," << Tmd(1, 3) << "," <<  yaw2 << ",";
        ofs << Tmd2(0, 3) << "," << Tmd2(1, 3) << "," << yaw3 << "," << gtvec[3] << "," << gtvec[4];
        ofs << "," << gtvec[5] << "," << gtvec[6] <<   "\n";

        i++;

        std::cout << gtvec[0] << "," << gtvec[1] << "," << gtvec[2] << std::endl;
        std::cout << T(0, 2) << "," << T(1, 2) << "," << yaw << std::endl;
        std::cout << Tmd(0, 3) << "," << Tmd(1, 3) << "," <<  yaw2 << std::endl;

        cv::Mat img_matches;
        cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("good", img_matches);
        cv::waitKey(0);
    }

    return 0;
}
