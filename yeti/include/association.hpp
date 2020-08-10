#pragma once
#include <math.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

/*!
   \brief Enforce orthogonality conditions on the given rotation matrix such that det(R) == 1 and R.tranpose() * R = I
   \param R The input rotation matrix either 2x2 or 3x3, will be overwritten with a slightly modified matrix to
   satisfy orthogonality conditions.
*/
template<typename T>
void enforce_orthogonality(Eigen::Matrix<T, -1, -1> &R) {
    if (R.cols() == 3) {
        const Eigen::Matrix<T, 3, 1> col1 = R.block(0, 1, 3, 1).normalized();
        const Eigen::Matrix<T, 3, 1> col2 = R.block(0, 2, 3, 1).normalized();
        const Eigen::Matrix<T, 3, 1> newcol0 = col1.cross(col2);
        const Eigen::Matrix<T, 3, 1> newcol1 = col2.cross(newcol0);
        R.block(0, 0, 3, 1) = newcol0;
        R.block(0, 1, 3, 1) = newcol1;
        R.block(0, 2, 3, 1) = col2;
    } else if (R.cols() == 2) {
        const T epsilon = 0.001;
        if (fabs(R(0, 0) - R(1, 1)) > epsilon || fabs(R(1, 0) + R(0, 1)) > epsilon) {
            std::cout << "ERROR: this is not a proper rigid transformation!" << std::endl;
        }
        T a = (R(0, 0) + R(1, 1)) / 2;
        T b = (-R(1, 0) + R(0, 1)) / 2;
        T sum = sqrt(pow(a, 2) + pow(b, 2));
        a /= sum;
        b /= sum;
        R(0, 0) = a; R(0, 1) = b;
        R(1, 0) = -b; R(1, 1) = a;
    }
}

/*!
   \brief Retrieve the rigid transformation that transforms points in p1 into points in p2.
   The output transform type (float or double) and size SE(2) vs. SE(3) depends on the size of the input points p1, p2.
   \param p1 A dim x N vector of points in either 2D (dim = 2) or 3D (dim = 3)
   \param p2 A dim x N vector of points in either 2D (dim = 2) or 3D (dim = 3)
   \param Tf [out] This matrix will be overwritten as the output transform
   \pre p1 and p2 are the same size. p1 and p2 are the matched feature point locations between two point clouds
   \post orthogonality is enforced on the rotation matrix.
*/
template<typename T>
void get_rigid_transform(Eigen::Matrix<T, -1, -1> p1, Eigen::Matrix<T, -1, -1> p2, Eigen::Matrix<T, -1, -1> &Tf) {
    assert(p1.cols() == p2.cols());
    assert(p1.rows() == p2.rows());
    const int dim = p1.rows();
    Eigen::Matrix<T, -1, -1> mu1 = Eigen::Matrix<T, -1, -1>::Zero(dim, 1);
    Eigen::Matrix<T, -1, -1> mu2 = mu1;
    // Calculate centroid of each point cloud
    for (int i = 0; i < p1.cols(); ++i) {
        mu1 += p1.block(0, i, dim, 1);
        mu2 += p2.block(0, i, dim, 1);
    }
    mu1 /= p1.cols();
    mu2 /= p1.cols();
    // Subtract centroid from each cloud
    auto q1 = p1;
    auto q2 = p2;
    for (int i = 0; i < p1.cols(); ++i) {
        q1.block(0, i, dim, 1) -= mu1;
        q2.block(0, i, dim, 1) -= mu2;
    }
    // Calculate rotation using SVD
    Eigen::Matrix<T, -1, -1> H = Eigen::Matrix<T, -1, -1>::Zero(dim, dim);
    for (int i = 0; i < p1.cols(); ++i) {
        Eigen::Matrix<T, -1, -1> H_prime = Eigen::Matrix<T, -1, -1>::Zero(dim, dim);
        H_prime = q1.block(0, i, dim, 1) * q2.block(0, i, dim, 1).transpose();
        H += H_prime;
    }
    auto svd = H.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<T, -1, -1> U = svd.matrixU();
    Eigen::Matrix<T, -1, -1> V = svd.matrixV();
    Eigen::Matrix<T, -1, -1> R_hat = V * U.transpose();
    if (R_hat.determinant() < 0) {
        V.block(0, dim - 1, dim, 1) = -1 * V.block(0, dim - 1, dim, 1);
        R_hat = V * U.transpose();
    }
    if (R_hat.determinant() != 1.0)
        enforce_orthogonality(R_hat);
    // Calculate translation
    Eigen::Matrix<T, -1, -1> t = mu2 - R_hat * mu1;
    // Create the output transformation
    Tf = Eigen::Matrix<T, -1, -1>::Identity(dim + 1, dim + 1);
    Tf.block(0, 0, dim, dim) = R_hat;
    Tf.block(0, dim, dim, 1) = t;
}

/*!
   \brief Returns a random subset of indices, where 0 <= indices[i] <= max_index. indices are non-repeating.
*/
static std::vector<int> random_subset(int max_index, int subset_size) {
    std::vector<int> subset;
    if (max_index < 0 || subset_size < 0)
        return subset;
    if (max_index < subset_size)
        subset_size = max_index;
    subset = std::vector<int>(subset_size, -1);
    for (uint i = 0; i < subset.size(); i++) {
        while (subset[i] < 0) {
            int idx = std::rand() % max_index;
            if (std::find(subset.begin(), subset.begin() + i, idx) == subset.begin() + i)
                subset[i] = idx;
        }
    }
    return subset;
}

template<class T>
class Ransac {
public:
    // p1, p2 need to be either (x, y) x N or (x, y, z) x N (must be in homogeneous coordinates)
    Ransac(Eigen::Matrix<T, -1, -1> p1_, Eigen::Matrix<T, -1, -1> p2_, float tolerance_, float inlier_ratio_,
        int iterations_) : p1(p1_), p2(p2_), tolerance(tolerance_), inlier_ratio(inlier_ratio_),
        iterations(iterations_) {
        assert(p1.cols() == p2.cols());
        assert(p1.rows() == p2.rows());
        int dim = p1.rows();
        assert(dim == 2 || dim == 3);
        T_best = Eigen::Matrix<T, -1, -1>::Identity(dim + 1, dim + 1);
    }
    void setTolerance(float tolerance_) {tolerance = tolerance_;}
    void setInlierRatio(float inlier_ratio_) {inlier_ratio = inlier_ratio_;}
    void setMaxIterations(int iterations_) {iterations = iterations_;}
    void getTransform(Eigen::Matrix<T, -1, -1> &Tf) {Tf = T_best;}

    int computeModel() {
        uint max_inliers = 0;
        std::vector<int> best_inliers;
        int dim = p1.rows();
        int subset_size = 2;
        int i = 0;
        for (i = 0; i < iterations; ++i) {
            std::vector<int> subset = random_subset(p1.cols(), subset_size);
            if ((int)subset.size() < subset_size)
                continue;
            // Compute transform from the random sample
            Eigen::Matrix<T, -1, -1> p1small, p2small;
            p1small = Eigen::Matrix<T, -1, -1>::Zero(dim, subset_size);
            p2small = p1small;
            for (int j = 0; j < subset_size; ++j) {
                p1small.block(0, j, dim, 1) = p1.block(0, subset[j], dim, 1);
                p2small.block(0, j, dim, 1) = p2.block(0, subset[j], dim, 1);
            }
            Eigen::Matrix<T, -1, -1> T_current;
            get_rigid_transform(p1small, p2small, T_current);
            // Check the number of inliers
            std::vector<int> inliers;
            getInliers(T_current, inliers);
            if (inliers.size() > max_inliers) {
                best_inliers = inliers;
                max_inliers = inliers.size();
            }
            if (float(inliers.size()) / float(p1.cols()) > inlier_ratio)
                break;
        }
        // Refine transformation using the inlier set
        Eigen::Matrix<T, -1, -1> p1small, p2small;
        p1small = Eigen::Matrix<T, -1, -1>::Zero(dim, best_inliers.size());
        p2small = p1small;
        for (uint j = 0; j < best_inliers.size(); ++j) {
            p1small.block(0, j, dim, 1) = p1.block(0, best_inliers[j], dim, 1);
            p2small.block(0, j, dim, 1) = p2.block(0, best_inliers[j], dim, 1);
        }
        get_rigid_transform(p1small, p2small, T_best);
        // std::cout << "iterations: " << i << std::endl;
        // std::cout << "inlier ratio: " << float(max_inliers) / float(p1.cols()) << std::endl;
        return max_inliers;
    }

private:
    Eigen::Matrix<T, -1, -1> p1, p2;
    float tolerance = 0.05;
    float inlier_ratio = 0.9;
    int iterations = 40;
    Eigen::Matrix<T, -1, -1> T_best;

    void getInliers(Eigen::Matrix<T, -1, -1> Tf, std::vector<int> &inliers) {
        int dim = p1.rows();
        Eigen::Matrix<T, -1, -1> p1_prime = Eigen::Matrix<T, -1, -1>::Ones(dim + 1, p1.cols());
        p1_prime.block(0, 0, dim, p1.cols()) = p1;
        p1_prime = Tf * p1_prime;
        inliers.clear();
        for (uint i = 0; i < p1_prime.cols(); ++i) {
            auto distance = (p1_prime.block(0, i, dim, 1) - p2.block(0, i, dim, 1)).norm();
            if (distance < tolerance)
                inliers.push_back(i);
        }
    }
};

Eigen::MatrixXf cross(Eigen::VectorXf x);

Eigen::MatrixXf circledot(Eigen::VectorXf x);

Eigen::Matrix4f se3ToSE3(Eigen::MatrixXf xi);


class MotionDistortedRansac {
public:
    MotionDistortedRansac(Eigen::MatrixXf p1, Eigen::MatrixXf p2, std::vector<float> a1_, std::vector<float> a2_,
        std::vector<int64_t> t1_, std::vector<int64_t> t2_, float tolerance_, float inlier_ratio_, int iterations_) :
        a1(a1_), a2(a2_), t1(t1_), t2(t2_), tolerance(tolerance_), inlier_ratio(inlier_ratio_),
        iterations(iterations_) {
        assert(p1.cols() == p2.cols());
        assert(p1.rows() == p2.rows());
        assert(p1.cols() >= p1.rows());
        dim = p1.rows();
        assert(dim == 2 || dim == 3);
        T_best = Eigen::MatrixXf::Identity(dim + 1, dim + 1);
        p1bar = Eigen::MatrixXf::Zero(4, p1.cols());
        p2bar = Eigen::MatrixXf::Zero(4, p2.cols());
        p1bar.block(3, 0, 1, p1.cols()) = Eigen::MatrixXf::Ones(1, p1.cols());
        p2bar.block(3, 0, 1, p2.cols()) = Eigen::MatrixXf::Ones(1, p2.cols());
        if (dim == 2) {
            p1bar.block(0, 0, 2, p1.cols()) = p1;
            p2bar.block(0, 0, 2, p2.cols()) = p2;
        } else if (dim == 3) {
            p1bar.block(0, 0, 3, p1.cols()) = p1;
            p2bar.block(0, 0, 3, p2.cols()) = p2;
        }
        R_pol << pow(0.25, 2), 0, 0, 0, 0, pow(0.0157, 2), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    }
    void setTolerance(float tolerance_) {tolerance = tolerance_;}
    void setInlierRatio(float inlier_ratio_) {inlier_ratio = inlier_ratio_;}
    void setMaxIterations(int iterations_) {iterations = iterations_;}
    void getTransform(Eigen::MatrixXf &Tf) {Tf = T_best;}
    int computeModel();

private:
    Eigen::MatrixXf p1bar, p2bar;
    std::vector<float> a1, a2;
    std::vector<int64_t> t1, t2;
    float tolerance = 0.05;
    float inlier_ratio = 0.9;
    int iterations = 40;
    int max_gn_iterations = 10;
    int dim = 2;
    Eigen::Matrix4f T_best = Eigen::Matrix4f::Identity();
    Eigen::VectorXf w_best = Eigen::VectorXf::Zero(6);
    Eigen::Matrix4f R_pol = Eigen::Matrix4f::Identity();
    Eigen::MatrixXf get_jacobian(Eigen::Vector3f gbar);
    Eigen::MatrixXf get_inv_jacobian(Eigen::Vector3f gbar);
    Eigen::VectorXf to_cylindrical(Eigen::VectorXf gbar);
    Eigen::VectorXf from_cylindrical(Eigen::VectorXf ybar);
    float get_delta_t(Eigen::VectorXf y2, Eigen::VectorXf y1);
    void get_motion_parameters(Eigen::MatrixXf &p1small, Eigen::MatrixXf &p2small, Eigen::VectorXf &wbar);
    void getInliers(Eigen::VectorXf wbar, std::vector<int> &inliers);
};
