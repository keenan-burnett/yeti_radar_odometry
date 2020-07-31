#pragma once
#include <math.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <iostream>

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
    std::cout << R_hat << std::endl;
    std::cout << R_hat.determinant() << std::endl;
    if (R_hat.determinant() < 0) {
        V.block(0, dim - 1, dim, 1) = -1 * V.block(0, dim - 1, dim, 1);
        R_hat = V * U.transpose();
    }
    if (R_hat.determinant() != 1.0)
        enforce_orthogonality(R_hat);
    std::cout << R_hat << std::endl;
    std::cout << R_hat.determinant() << std::endl;
    // Calculate translation
    Eigen::Matrix<T, -1, -1> t = mu2 - R_hat * mu1;
    // Create the output transformation
    Tf = Eigen::Matrix<T, -1, -1>::Identity(dim + 1, dim + 1);
    Tf.block(0, 0, dim, dim) = R_hat;
    Tf.block(0, dim, dim, 1) = t;
}
