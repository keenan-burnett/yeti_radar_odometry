#include <association.hpp>

Eigen::Matrix3f cross(Eigen::Vector3f x) {
    Eigen::Matrix3f X = {0, -x[3], x[2], x[3], 0, -x[1], -x[2], x[1], 0};
    return X;
}

void se3ToSE3(Eigen::MatrixXf xi, Eigen::Matrix4f &T) {
    assert(xi.rows() == 6);
    T = Eigen::Matrix4f::Identity();
    Eigen::VectorXf rho;
    Eigen::VectorXf phibar;
    rho = xi.block(0, 0, 3, 1);
    phibar = xi.block(0, 3, 3, 1);
    float phi = phibar.norm();
    phibar.normalize();
    Eigen::MatrixXf C = Eigen::MatrixXf::Identity(3);
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(3);
    C = cos(phi) * I + (1 - cos(phi)) * phibar * phibar.transpose() + sin(phi) * cross(phibar);
    Eigen::MatrixXf J = Eigen::MatrixXf::Identity(3);
    J = I * sin(phi) / phi + (1 - sin(phi) / phi) * phibar * phibar.transpose() + cross(phibar) * (1 - cos(phi)) / phi;
    Eigen::VectorXf r = J * rho;
    T.block(0, 0, 3, 3) = C;
    T.block(0, 3, 3, 1) = r;
}

Eigen::MatrixXf MotionDistortedRansac::get_jacobian(Eigen::MatrixXf gbar) { }

MotionDistortedRansac::get_motion_parameters(Eigen::MatrixXf p1small, Eigen::MatrixXf p2small, Eigen::VectorXf &wbar) {
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6, 6);
    Eigen::MatrixXf b = Eigen::MatrixXf::Zero(6, 1);

    for (int it = 0; it < max_gn_iterations; ++it) {
        for (int m = 0; m < p1small.cols(); ++m) {
            F = get_jacobian(gbar);
            H = F * G;
            A += H.tranpose() * R2.inverse() * H;
            b += H.transpose() * R2.inverse() * e;
        }
        delta_w = A.inverse() * b;
        wbar += delta_w;
        // TODO(keenan): Check for convergence
    }
}

MotionDistortedRansac::getInliers(Eigen::VectorXf wbar, std::vector<int> &inliers) { }

MotionDistortedRansac::computeModel() {
    uint max_inliers = 0;
    std::vector<int> best_inliers;
    Eigen::VectorXf wbar = Eigen::MatrixXf::Zero(6, 1);
    for (int i = 0; i < iterations; ++i) {
        std::vector<int> subset = random_subset(p1.cols(), dim);
        // NLLS to obtain the motion estimate
        Eigen::MatrixXf p1small, p2small;
        p1small = Eigen::MatrixXf::Zero(4, dim);
        p2small = p1small;
        for (int j = 0; j < dim; ++j) {
            p1small.block(0, j, 4, 1) = p1.block(0, subset[j], 4, 1);
            p2small.block(0, j, 4, 1) = p2.block(0, subset[j], 4, 1);
        }
        get_motion_parameters(p1small, p2small, wbar);
        // Check the number of inliers
        std::vector<int> inliers;
        getInliers(wbar, inliers);
        if (inliers.size() > max_inliers) {
            best_inliers = inliers;
            max_inliers = inliers.size();
        }
        if (float(inliers.size()) / float(p1.cols()) > inlier_ratio)
            break;
    }
    // Refine transformation using the inlier set
    Eigen::MatrixXf p1small, p2small;
    p1small = Eigen::MatrixXf::Zero(4, best_inliers.size());
    p2small = p1small;
    for (int j = 0; j < best_inliers.size(); ++j) {
        p1small.block(0, j, 4, 1) = p1.block(0, best_inliers[j], 4, 1);
        p2small.block(0, j, 4, 1) = p2.block(0, best_inliers[j], 4, 1);
    }
    get_motion_parameters(p1small, p2small, w_best);
    // std::cout << "iterations: " << i << std::endl;
    // std::cout << "inlier ratio: " << float(max_inliers) / float(p1.cols()) << std::endl;
    return max_inliers;
}
