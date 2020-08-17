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
void enforce_orthogonality(Eigen::MatrixXd &R);

/*!
   \brief Retrieve the rigid transformation that transforms points in p1 into points in p2.
   The output transform type (float or double) and size SE(2) vs. SE(3) depends on the size of the input points p1, p2.
   \param p1 A dim x N vector of points in either 2D (dim = 2) or 3D (dim = 3)
   \param p2 A dim x N vector of points in either 2D (dim = 2) or 3D (dim = 3)
   \param Tf [out] This matrix will be overwritten as the output transform
   \pre p1 and p2 are the same size. p1 and p2 are the matched feature point locations between two point clouds
   \post orthogonality is enforced on the rotation matrix.
*/
void get_rigid_transform(Eigen::MatrixXd p1, Eigen::MatrixXd p2, Eigen::MatrixXd &Tf);

/*!
   \brief Returns a random subset of indices, where 0 <= indices[i] <= max_index. indices are non-repeating.
*/
std::vector<int> random_subset(int max_index, int subset_size);

Eigen::MatrixXd cross(Eigen::VectorXd x);

Eigen::MatrixXd circledot(Eigen::VectorXd x);

Eigen::Matrix4d se3ToSE3(Eigen::MatrixXd xi);

Eigen::VectorXd SE3tose3(Eigen::MatrixXd T);

Eigen::MatrixXd eulerToRot(Eigen::VectorXd eul);

double wrapto2pi(double theta);

class Ransac {
public:
    // p1, p2 need to be either (x, y) x N or (x, y, z) x N (must be in homogeneous coordinates)
    Ransac(Eigen::MatrixXd p1_, Eigen::MatrixXd p2_, double tolerance_, double inlier_ratio_,
        int iterations_) : p1(p1_), p2(p2_), tolerance(tolerance_), inlier_ratio(inlier_ratio_),
        iterations(iterations_) {
        int dim = p1.rows();
        assert(p1.cols() == p2.cols() && p1.rows() == p2.rows() && (dim == 2 || dim == 3));
        T_best = Eigen::MatrixXd::Identity(dim + 1, dim + 1);
    }
    void setTolerance(double tolerance_) {tolerance = tolerance_;}
    void setInlierRatio(double inlier_ratio_) {inlier_ratio = inlier_ratio_;}
    void setMaxIterations(int iterations_) {iterations = iterations_;}
    void getTransform(Eigen::MatrixXd &Tf) {Tf = T_best;}

    int computeModel();

private:
    Eigen::MatrixXd p1, p2;
    double tolerance = 0.05;
    double inlier_ratio = 0.9;
    int iterations = 40;
    Eigen::MatrixXd T_best;

    void getInliers(Eigen::MatrixXd Tf, std::vector<int> &inliers);
};

// All operations are done in SE(3) even if the input is 2D. The output motion and transforms are in 3D.
class MotionDistortedRansac {
public:
    MotionDistortedRansac(Eigen::MatrixXd p1, Eigen::MatrixXd p2, std::vector<int64_t> t1_, std::vector<int64_t> t2_,
        double tolerance_, double inlier_ratio_, int iterations_) : t1(t1_), t2(t2_), tolerance(tolerance_),
        inlier_ratio(inlier_ratio_), iterations(iterations_) {
        const int dim = p1.rows();
        assert(p1.cols() == p2.cols() && p1.rows() == p2.rows() && p1.cols() >= p1.rows() && (dim == 2 || dim == 3));
        T_best = Eigen::MatrixXd::Zero(dim + 1, dim + 1);
        p1bar = Eigen::MatrixXd::Zero(4, p1.cols());
        p2bar = Eigen::MatrixXd::Zero(4, p2.cols());
        p1bar.block(3, 0, 1, p1.cols()) = Eigen::MatrixXd::Ones(1, p1.cols());
        p2bar.block(3, 0, 1, p2.cols()) = Eigen::MatrixXd::Ones(1, p2.cols());
        if (dim == 2) {
            p1bar.block(0, 0, 2, p1.cols()) = p1;
            p2bar.block(0, 0, 2, p2.cols()) = p2;
        } else if (dim == 3) {
            p1bar.block(0, 0, 3, p1.cols()) = p1;
            p2bar.block(0, 0, 3, p2.cols()) = p2;
        }
        R_pol << pow(0.25, 2), 0, 0, 0, 0, pow(0.0157, 2), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
        y1bar = Eigen::MatrixXd::Zero(4, p1.cols());
        y2bar = Eigen::MatrixXd::Zero(4, p1.cols());
        delta_ts = std::vector<double>(p1.cols(), 0.0);
        for (uint i = 0; i < p1bar.cols(); ++i) {
            y1bar.block(0, i, 4, 1) = to_cylindrical(p1bar.block(0, i, 4, 1));
            y2bar.block(0, i, 4, 1) = to_cylindrical(p2bar.block(0, i, 4, 1));
            int64_t delta_t = t2[i] - t1[i];
            delta_ts[i] = double(delta_t) / 1000000.0;
        }
    }
    void setTolerance(double tolerance_) {tolerance = tolerance_;}
    void setInlierRatio(double inlier_ratio_) {inlier_ratio = inlier_ratio_;}
    void setMaxIterations(int iterations_) {iterations = iterations_;}
    void setMaxGNIterations(int iterations_) {max_gn_iterations = iterations_;}
    void setConvergenceThreshold(double eps) {epsilon_converge = eps;}
    void getTransform(Eigen::MatrixXd &Tf) {Tf = T_best;}
    void getMotion(Eigen::VectorXd &w) {w = w_best;}
    int computeModel();

private:
    Eigen::MatrixXd p1bar, p2bar;
    Eigen::MatrixXd y1bar, y2bar;
    std::vector<double> delta_ts;
    std::vector<int64_t> t1, t2;
    double tolerance = 0.05;
    double inlier_ratio = 0.9;
    int iterations = 40;
    int max_gn_iterations = 10;
    double epsilon_converge = 0.01;
    int dim = 2;
    Eigen::MatrixXd T_best;
    Eigen::VectorXd w_best = Eigen::VectorXd::Zero(6);
    Eigen::Matrix4d R_pol = Eigen::Matrix4d::Identity();
    Eigen::MatrixXd get_jacobian(Eigen::Vector4d gbar);
    Eigen::MatrixXd get_inv_jacobian(Eigen::Vector4d gbar);
    Eigen::VectorXd to_cylindrical(Eigen::VectorXd gbar);
    Eigen::VectorXd from_cylindrical(Eigen::VectorXd ybar);
    void get_motion_parameters(Eigen::MatrixXd &p1small, Eigen::MatrixXd &p2small, std::vector<double> delta_t_local,
        Eigen::VectorXd &wbar);
    void get_motion_parameters2(Eigen::MatrixXd& p1small, Eigen::MatrixXd& p2small, std::vector<double> delta_t_local,
        Eigen::VectorXd &wbar);
    void getInliers(Eigen::VectorXd wbar, std::vector<int> &inliers);
};
