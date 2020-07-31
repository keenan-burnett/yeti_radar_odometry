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


int main(int argc, char *argv[]) {
    std::string datadir = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/radar";
    std::string gt = "/home/keenan/Documents/data/2019-01-10-14-36-48-radar-oxford-10k-partial/gt/radar_odometry.csv";
    float cart_resolution = 0.25;
    int cart_pixel_width = 1000;
    int min_range = 58;
    float radar_resolution = 0.0432;
    // int max_points = 10000;
    bool interp = true;
    float zq = 3.0;
    int sigma_gauss = 17;
    int patch_size = 21;
    // Get file names of the radar images
    std::vector<std::string> radar_files;
    get_file_names(datadir, radar_files);
    // File for storing the results of ICP on each frame (and the accuracy)
    std::string accfile = "accuracy.csv";
    std::ofstream ofs;
    ofs.open(accfile, std::ios::out);
    ofs << "x,y,yaw,gtx,gty,gtyaw,time1,time2\n";
    // Create ORB feature detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));

    cv::Mat img1, img2, desc1, desc2;
    std::vector<cv::KeyPoint> kp1, kp2;
    std::vector<int64_t> t1, t2;
    std::vector<float> a1, a2;
    std::vector<bool> v1, v2;
    cv::Mat f1, f2;
    Eigen::MatrixXf cart_targets1, cart_targets2;

    for (uint i = 0; i < radar_files.size() - 1; ++i) {
        if (i == 0) {
            // Read the two radar images
            load_radar(datadir + "/" + radar_files[i], t1, a1, v1, f1);
            // Extract features locations
            Eigen::MatrixXf targets;
            cen2018features(f1, zq, sigma_gauss, min_range, targets);
            // Convert targets to cartesian coordinates
            polar_to_cartesian_points(a1, targets, radar_resolution, cart_targets1);
            // Convert radar targets into opencv keypoints
            std::vector<cv::Point2f> bev_points;
            convert_to_bev(cart_targets1, cart_resolution, cart_pixel_width, bev_points);
            kp1.clear();
            for (uint j = 0; j < bev_points.size(); ++j) {
                kp1.push_back(cv::KeyPoint(bev_points[j], patch_size));
            }
            // Compute ORB feature descriptors
            radar_polar_to_cartesian(a1, f1, radar_resolution, cart_resolution, cart_pixel_width, interp, img1);
            double min, max;
            cv::minMaxLoc(img1, &min, &max);
            img1.convertTo(img1, CV_8UC1, 255.0 / max);
            detector->compute(img1, kp1, desc1);
        } else {
            t1 = t2; a1 = a2; v1 = v2; f1 = f2; kp1 = kp2; cart_targets1 = cart_targets2;
            desc1 = desc2.clone(); img2.copyTo(img1);
        }

        load_radar(datadir + "/" + radar_files[i + 1], t2, a2, v2, f2);
        Eigen::MatrixXf targets2;
        cen2018features(f2, zq, sigma_gauss, min_range, targets2);
        polar_to_cartesian_points(a2, targets2, radar_resolution, cart_targets2);
        std::vector<cv::Point2f> bev_points2;
        convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, bev_points2);
        kp2.clear();
        for (uint j = 0; j < bev_points2.size(); ++j) {
            kp2.push_back(cv::KeyPoint(bev_points2[j], patch_size));
        }
        radar_polar_to_cartesian(a2, f2, radar_resolution, cart_resolution, cart_pixel_width, interp, img2);
        double min, max;
        cv::minMaxLoc(img2, &min, &max);
        img2.convertTo(img2, CV_8UC1, 255.0 / max);
        detector->compute(img2, kp2, desc2);

        // Match keypoint descriptors using FLANN
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(desc1, desc2, knn_matches, 2);
        // Filter matches using nearest neighbor distance ratio (Szeliski)
        const float ratio = 0.7;
        std::vector<cv::DMatch> good_matches;
        std::cout << knn_matches.size() << std::endl;
        for (uint j = 0; j < knn_matches.size(); ++j) {
            if (!knn_matches[j].size())
                continue;
            if (knn_matches[j][0].distance < ratio * knn_matches[j][1].distance) {
                std::cout << j << " " << knn_matches[j][0].trainIdx << " " << knn_matches[j][0].queryIdx << std::endl;
                good_matches.push_back(knn_matches[j][0]);
            }
        }
        std::cout << good_matches.size() << std::endl;
        cv::Mat img_matches;
        cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>());

        cv::imshow("good", img_matches);
        cv::waitKey(0);

        // Compute initial transformation using RANSAC

        // Refine transformation using the inlier set

        // Retrieve the ground truth to calculate accuracy
        // std::vector<std::string> parts;
        // boost::split(parts, radar_files[i], boost::is_any_of("."));
        // int64 time1 = std::stoll(parts[0]);
        // boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        // int64 time2 = std::stoll(parts[0]);
        // std::vector<float> gtvec;
        // get_groundtruth_odometry(gt, time1, time2, gtvec);
        // float yaw = -1 * asin(T(0, 1));
        // // Write estimated and ground truth transform to the csv file
        // ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ",";
        // ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[5] << ",";
        // ofs << time1 << "," << time2 << "\n";
    }

    return 0;
}
