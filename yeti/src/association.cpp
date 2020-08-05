#include <association.hpp>

// template<class T>
// void Ransac::getInliers(Eigen::Matrix<T, -1, -1> Tf, std::vector<int> &inliers) {
//     Eigen::Matrix<T, -1, -1> p1_prime = Tf * p1;
//     int dim = p1.rows();
//     inliers.clear();
//     for (uint i = 0; i < p1_prime.cols(); ++i) {
//         auto distance = (p1_prime.block(0, i, dim, 1) - p2.block(0, i, dim, 1)).norm();
//         if (distance < tolerance)
//             inliers.push_back(i);
//     }
// }



// template<class T>
// int Ransac::computeModel() {
//     uint max_inliers = 0;
//     int dim = p1.rows();
//     for (int i = 0; i < iterations; ++i) {
//         std::vector<int> subset = random_subset(p1.cols(), 2);
//         if (subset.size() < 2)
//             continue;
//         // check for colinearity
//         Eigen::Matrix<T, -1, -1> a = p1.block(0, subset[0], dim, 1) - p1.block(0, subset[1], dim, 1);
//         Eigen::Matrix<T, -1, -1> b = p2.block(0, subset[0], dim, 1) - p2.block(0, subset[1], dim, 1);
//         float cos_theta = a.dot(b) / (a.norm() * b.norm());
//         if (cos_theta > colinear_angle_threshold)
//             continue;
//         // compute transform from the random sample
//         Eigen::Matrix<T, dim, 2> p1small, p2small;
//         p1small.block(0, 0, dim, 1) = p1.block(0, subset[0], dim, 1);
//         p1small.block(0, 1, dim, 1) = p1.block(0, subset[1], dim, 1);
//         p2small.block(0, 0, dim, 1) = p2.block(0, subset[0], dim, 1);
//         p2small.block(0, 1, dim, 1) = p2.block(0, subset[1], dim, 1);
//         Eigen::Matrix<T, -1, -1> T_current;
//         get_rigid_transform(p1small, p2small, T_current);
//         std::vector<int> inliers;
//         getInliers(T_current, inliers);
//         if (inliers.size() > max_inliers) {
//             T_best = T_current;
//             max_inliers = inliers.size();
//         }
//         if (float(num_inliers) / float(p1.cols()) > inlier_ratio)
//             break;
//     }
// }
